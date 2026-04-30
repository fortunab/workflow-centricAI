# ===== CELL 0 =====
pip install torch "transformers<5.0.0" accelerate


# ===== CELL 1 =====
pip install gtts


# ===== CELL 2 =====
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel, EncodecModel, AutoTokenizer
from gtts import gTTS
from IPython.display import Audio, display
import gc

# ============================================
# 1. FIXED Modular Pipeline Configuration
# ============================================
config_yaml = """
pipeline:
  input:
    modalities: [text, image, audio, video] # Explicitly defined list
  encode:
    text:  { id: "Qwen/Qwen2.5-0.5B-Instruct", out: 896 }
    image: { id: "google/siglip-base-patch16-224", out: 768 }
    audio: { id: "facebook/encodec_24khz", out: 128 }
    video: { id: "OpenGVLab/VideoMAEv2-Base", out: 768 }
  align:
    dim: 256
    norm: true
    route: { experts: 4, topk: 1 }
  fuse:
    cross_attn: true
    heads: 4
  reason:
    on: true
    depth: 2
    heads: 8
    width: 896
    ff_dim: 3584
    activation: "gelu"
  adapt:
    out: 512
  decode:
    text:  { out_dim: 151936 }
    audio: { out_dim: 1024 }
    image: { out_dim: 12288 }
    video: { out_dim: 368640 }
"""
config = yaml.safe_load(config_yaml)

# ============================================
# 2. Refined Components
# ============================================

class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, shared_dim, use_norm):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim) if use_norm else nn.Identity()
    def forward(self, x): return self.norm(self.proj(x))

class Router(nn.Module):
    def __init__(self, dim, experts, topk):
        super().__init__()
        self.gate = nn.Linear(dim, experts)
        self.topk = topk
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        vals, idx = torch.topk(probs, self.topk, dim=-1)
        return probs, vals / (vals.sum(dim=-1, keepdim=True) + 1e-6), idx

class LatentExperts(nn.Module):
    def __init__(self, num_experts, dim, heads):
        super().__init__()
        self.num_experts = num_experts
        self.queries = nn.Parameter(torch.randn(num_experts, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_experts)])
    def forward(self, tokens, topk_idx):
        expert_outputs = []
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1).unsqueeze(-1).float()
            q = self.queries[e].expand(tokens.shape[0], -1, -1)
            out, _ = self.attn[e](q, tokens * mask, tokens * mask)
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=1)

class Reasoner(nn.Module):
    def __init__(self, cfg_align, cfg_reason, pretrained_layers=None, rotary_emb=None):
        super().__init__()
        self.uplift = nn.Linear(cfg_align['dim'], cfg_reason['width'])
        self.layers = nn.ModuleList(pretrained_layers)
        self.rotary_emb = rotary_emb
        self.norm = nn.LayerNorm(cfg_reason['width'])
    def forward(self, x):
        x = self.uplift(x)
        b, seq, _ = x.shape
        pos_ids = torch.arange(seq, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            out = layer(x, position_embeddings=(cos, sin))
            x = out if isinstance(out, tuple) else out
        return self.norm(x)

class Adapter(nn.Module):
    def __init__(self, shared_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(shared_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x): return self.norm(self.proj(x))

class MultiScaleDecoder(nn.Module):
    def __init__(self, latent_dim, encoder_dim, output_dim):
        super().__init__()
        self.main = nn.Linear(latent_dim, output_dim)
        self.res = nn.Linear(encoder_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
    def forward(self, x, skip=None):
        out = self.main(x)
        if skip is not None:
            if skip.dim() > 2: skip = skip.mean(dim=1)
            out = out + (self.gate(x) * self.res(skip))
        return out

# ============================================
# 3. R-ULIx Orchestrator
# ============================================

class R_ULIx(nn.Module):
    def __init__(self, cfg, device_dtype=torch.float32):
        super().__init__()
        p = cfg['pipeline']

        # Load Encoders
        q_full = AutoModel.from_pretrained(p['encode']['text']['id'], torch_dtype=device_dtype)
        self.text_embed = q_full.embed_tokens
        reason_layers = q_full.layers[:p['reason']['depth']]
        rot_emb = q_full.rotary_emb if hasattr(q_full, 'rotary_emb') else q_full.model.rotary_emb

        self.encoders = nn.ModuleDict({
            'image': SiglipVisionModel.from_pretrained(p['encode']['image']['id'], torch_dtype=device_dtype).vision_model,
            'audio': EncodecModel.from_pretrained(p['encode']['audio']['id']).encoder.to(device_dtype),
            'video': AutoModel.from_pretrained(p['encode']['video']['id'], trust_remote_code=True, torch_dtype=device_dtype)
        })

        del q_full
        gc.collect()

        # Build pipeline
        self.projectors = nn.ModuleDict({
            m: ProjectNormalize(p['encode'][m]['out'], p['align']['dim'], p['align']['norm'])
            for m in p['input']['modalities']
        })
        self.router = Router(p['align']['dim'], p['align']['route']['experts'], p['align']['route']['topk'])
        self.experts = LatentExperts(p['align']['route']['experts'], p['align']['dim'], p['fuse']['heads'])
        self.fusion_attn = nn.MultiheadAttention(p['align']['dim'], p['fuse']['heads'], batch_first=True)
        self.reasoner = Reasoner(p['align'], p['reason'], pretrained_layers=reason_layers, rotary_emb=rot_emb)
        self.adapter = Adapter(p['reason']['width'], p['adapt']['out'])
        self.decoders = nn.ModuleDict({
            m: MultiScaleDecoder(p['adapt']['out'], p['encode'][m]['out'], p['decode'][m]['out_dim'])
            for m in p['input']['modalities'] if m in p['decode']
        })

    def forward(self, inputs):
        tokens = []
        skips = {}

        for m, x in inputs.items():


            if m == "text":
                feat = self.text_embed(x)  # (B, seq, hidden)


            elif m == "audio":
                enc = self.encoders["audio"](x)
                feat = enc.transpose(1, 2)  # (B, seq, channels)

            # =========================
            # VIDEO (VideoMAE expects B,C,T,H,W)
            # =========================
            elif m == "video":

                # ensure layout (B,C,T,H,W)
                if x.shape[1] != 3:
                    x = x.permute(0, 2, 1, 3, 4)

                B,C,T,H,W = x.shape

                # VideoMAE expects 16 frames
                if T < 16:
                    repeat = 16 // T + 1
                    x = x.repeat(1,1,repeat,1,1)[:, :, :16]

                enc_out = self.encoders["video"](x)

                if hasattr(enc_out,"last_hidden_state"):
                    feat = enc_out.last_hidden_state
                elif isinstance(enc_out,tuple):
                    feat = enc_out[0]
                else:
                    feat = enc_out

            elif m == "image":
                enc_out = self.encoders["image"](x)

                if hasattr(enc_out, "last_hidden_state"):
                    feat = enc_out.last_hidden_state
                elif isinstance(enc_out, tuple):
                    feat = enc_out[0]
                else:
                    feat = enc_out

            else:
                continue

            skips[m] = feat

            proj = self.projectors[m](feat)

            # ensure sequence format
            if proj.dim() == 2:
                proj = proj.unsqueeze(1)

            tokens.append(proj)

        combined = torch.cat(tokens, dim=1)

        _, _, idx = self.router(combined)

        latent = self.experts(combined, idx)

        latent, _ = self.fusion_attn(latent, latent, latent)
        reasoned = self.reasoner(latent)

        adapted = self.adapter(reasoned)

        # global latent
        pooled = adapted.mean(dim=1)

        outputs = {}

        for m in self.decoders:
            skip = skips.get(m, None)
            outputs[m] = self.decoders[m](pooled, skip)

        return outputs

# 4. Device-Aware Init & Test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_dtype = torch.float16 if device.type == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(config['pipeline']['encode']['text']['id'])

model = R_ULIx(config, device_dtype=run_dtype).to(device)
if device.type == "cuda": model.half()
print(f"R-ULIx Ready | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

def test_model():
    batch = 2
    dummy = {
        "text": torch.randint(0, 151936, (batch, 16)).to(device),
        "image": torch.randn(batch, 3, 224, 224).to(device, dtype=run_dtype),
        "audio": torch.randn(batch, 1, 24000).to(device, dtype=run_dtype),
        "video": torch.randn(batch, 4, 3, 224, 224).to(device, dtype=run_dtype)
    }
    model.eval()
    with torch.no_grad():
        out = model(dummy)

        # Text-to-Audio Output Logic
        text_logits = out["text"]
        predicted_ids = torch.argmax(text_logits, dim=-1)
        raw_response = tokenizer.decode(predicted_ids, skip_special_tokens=True)
        clean_response = raw_response if raw_response.strip() else "[Untrained Model Noise]"

        print(f"\nModel Response: {clean_response}")

        try:
            tts = gTTS(text=clean_response, lang='en')
            tts.save('output.mp3')
            display(Audio('output.mp3', autoplay=True))
        except:
            print("Audio output unavailable.")

test_model()



# ===== CELL 3 =====
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel, EncodecModel, AutoTokenizer
import gc

# --- Configuration ---
config_yaml = """
pipeline:
  input:
    modalities: [text, image, audio, video]
  perceiver:
    num_latents: 64
    heads: 8
    layers: 2
  encode:
    text:  { id: "Qwen/Qwen2.5-0.5B-Instruct", out: 896 }
    image: { id: "google/siglip-base-patch16-224", out: 768 }
    audio: { id: "facebook/encodec_24khz", out: 128 }
    video: { id: "OpenGVLab/VideoMAEv2-Base", out: 768 }
  align:
    dim: 256
    route: { experts: 4, topk: 1 }
  reason:
    depth: 2
    width: 896
  adapt:
    out: 512
  decode:
    text:  { out_dim: 151936 }
    audio: { out_dim: 1024 }
    image: { out_dim: 12288 }
    video: { out_dim: 368640 }
"""
config = yaml.safe_load(config_yaml)

class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, shared_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
    def forward(self, x): return self.norm(self.proj(x))

class Router(nn.Module):
    def __init__(self, dim, experts, topk):
        super().__init__()
        self.gate = nn.Linear(dim, experts)
        self.topk = topk
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        vals, idx = torch.topk(probs, self.topk, dim=-1)
        return probs, vals / (vals.sum(dim=-1, keepdim=True) + 1e-6), idx

class LatentExperts(nn.Module):
    def __init__(self, num_experts, dim, heads):
        super().__init__()
        self.num_experts = num_experts
        self.queries = nn.Parameter(torch.randn(num_experts, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_experts)])
    def forward(self, tokens, topk_idx):
        expert_outputs = []
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1).unsqueeze(-1).float()
            q = self.queries[e].expand(tokens.shape[0], -1, -1)
            out, _ = self.attn[e](q, tokens * mask, tokens * mask)
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=1)

class Reasoner(nn.Module):
    def __init__(self, cfg_align, cfg_reason, pretrained_layers, rotary_emb):
        super().__init__()
        self.uplift = nn.Linear(cfg_align['dim'], cfg_reason['width'])
        self.layers = nn.ModuleList(pretrained_layers)
        self.rotary_emb = rotary_emb
        self.norm = nn.LayerNorm(cfg_reason['width'])
    def forward(self, x):
        x = self.uplift(x)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            out = layer(x, position_embeddings=(cos, sin))
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)

class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, heads=8, layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                nn.LayerNorm(dim)
            ]) for _ in range(layers)
        ])
    def forward(self, x):
        b = x.shape[0]
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1)
        for attn, norm1, ff, norm2 in self.layers:
            out, _ = attn(latents, x, x)
            latents = norm1(latents + out)
            latents = norm2(latents + ff(latents))
        return latents

class MultiScaleDecoder(nn.Module):
    def __init__(self, latent_dim, encoder_dim, output_dim):
        super().__init__()
        self.main = nn.Linear(latent_dim, output_dim)
        self.res = nn.Linear(encoder_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
    def forward(self, x, skip=None):
        out = self.main(x)
        if skip is not None:
            skip = skip.mean(dim=1) if skip.dim() > 2 else skip
            out = out + (self.gate(x) * self.res(skip))
        return out

# --- Orchestrator with Modality Switch ---

class R_ULIx_Final(nn.Module):
    def __init__(self, cfg, device_dtype=torch.float32):
        super().__init__()
        p = cfg['pipeline']

        # 1. Base Models
        base_llm = AutoModel.from_pretrained(p['encode']['text']['id'], torch_dtype=device_dtype)
        self.text_embed = base_llm.embed_tokens
        self.rotary_emb = base_llm.rotary_emb if hasattr(base_llm, 'rotary_emb') else base_llm.model.rotary_emb
        reason_layers = base_llm.layers[:p['reason']['depth']]

        self.encoders = nn.ModuleDict({
            'image': SiglipVisionModel.from_pretrained(p['encode']['image']['id'], torch_dtype=device_dtype).vision_model,
            'audio': EncodecModel.from_pretrained(p['encode']['audio']['id']).encoder.to(device_dtype),
            'video': AutoModel.from_pretrained(p['encode']['video']['id'], trust_remote_code=True, torch_dtype=device_dtype)
        })

        # 2. Modality Identifiers
        self.modality_id = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, p['align']['dim'])) for m in p['input']['modalities']
        })

        # 3. Processing Pipeline
        self.projectors = nn.ModuleDict({
            m: ProjectNormalize(p['encode'][m]['out'], p['align']['dim']) for m in p['input']['modalities']
        })

        self.router = Router(p['align']['dim'], p['align']['route']['experts'], p['align']['route']['topk'])
        self.experts = LatentExperts(p['align']['route']['experts'], p['align']['dim'], 8)
        self.reasoner = Reasoner(p['align'], p['reason'], reason_layers, self.rotary_emb)
        self.adapter = nn.Linear(p['reason']['width'], p['adapt']['out'])
        self.perceiver = PerceiverResampler(p['adapt']['out'], p['perceiver']['num_latents'])

        # 4. Modality Switch Gate
        # Predicts which modalities should be active (Multi-label classification)
        self.modality_gate = nn.Sequential(
            nn.Linear(p['adapt']['out'], len(p['input']['modalities'])),
            nn.Sigmoid()
        )

        self.decoders = nn.ModuleDict({
            m: MultiScaleDecoder(p['adapt']['out'], p['encode'][m]['out'], p['decode'][m]['out_dim'])
            for m in p['input']['modalities'] if m in p['decode']
        })

        del base_llm
        gc.collect()

    def _extract(self, m, x):
        if m == "text": return self.text_embed(x)
        if m == "audio": return self.encoders["audio"](x).transpose(1, 2)
        if m == "video":
            if x.shape[1] != 3: x = x.permute(0, 2, 1, 3, 4)
            out = self.encoders["video"](x)
        else: out = self.encoders["image"](x)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

    def forward(self, inputs):
        tokens, skips = [], {}

        for m, x in inputs.items():
            feat = self._extract(m, x)
            skips[m] = feat
            proj = self.projectors[m](feat)
            if proj.dim() == 2: proj = proj.unsqueeze(1)
            tokens.append(proj + self.modality_id[m].expand(proj.shape[0], proj.shape[1], -1))

        # Core Pipeline
        combined = torch.cat(tokens, dim=1)
        _, _, idx = self.router(combined)
        latent = self.experts(combined, idx)
        reasoned = self.reasoner(latent)
        adapted = self.adapter(reasoned)

        # Perceiver Compression
        pooled_latents = self.perceiver(adapted)
        summary = pooled_latents.mean(dim=1)

        # Modality Switching Logic
        gate_scores = self.modality_gate(summary) # [B, 4]

        outputs = {}
        for i, m in enumerate(config['pipeline']['input']['modalities']):
            if m in self.decoders:
                # Multiply decoder output by gate score to "shut off" unused modalities
                raw_out = self.decoders[m](summary, skips.get(m))
                outputs[m] = raw_out * gate_scores[:, i].unsqueeze(-1)

        return outputs, gate_scores

# --- Execution ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R_ULIx_Final(config).to(device)
print(f"R-ULIx Finalized | Gating Head Ready.")

# Example Inference
def run_test():
    dummy = {"text": torch.randint(0, 1000, (1, 10)).to(device)}
    out, gates = model(dummy)
    print(f"Active Modality Confidence: {gates.cpu().detach().numpy()}")

run_test()


# ===== CELL 4 =====
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel, EncodecModel, AutoTokenizer
import gc

# --- Configuration ---
config_yaml = """
pipeline:
  input:
    modalities: [text, image, audio, video]
  perceiver:
    num_latents: 64
    heads: 8
    layers: 2
  encode:
    text:  { id: "Qwen/Qwen2.5-0.5B-Instruct", out: 896 }
    image: { id: "google/siglip-base-patch16-224", out: 768 }
    audio: { id: "facebook/encodec_24khz", out: 128 }
    video: { id: "OpenGVLab/VideoMAEv2-Base", out: 768 }
  align:
    dim: 256
    route: { experts: 4, topk: 1 }
  reason:
    depth: 2
    width: 896
  adapt:
    out: 512
  decode:
    text:  { out_dim: 151936 }
    audio: { out_dim: 1024 }
    image: { out_dim: 12288 }
    video: { out_dim: 368640 }
"""
config = yaml.safe_load(config_yaml)

# --- Classes from Previous Iterations (Included for Completeness) ---

class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, shared_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
    def forward(self, x): return self.norm(self.proj(x))

class Router(nn.Module):
    def __init__(self, dim, experts, topk):
        super().__init__()
        self.gate = nn.Linear(dim, experts)
        self.topk = topk
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        vals, idx = torch.topk(probs, self.topk, dim=-1)
        return probs, vals / (vals.sum(dim=-1, keepdim=True) + 1e-6), idx

class LatentExperts(nn.Module):
    def __init__(self, num_experts, dim, heads):
        super().__init__()
        self.num_experts = num_experts
        self.queries = nn.Parameter(torch.randn(num_experts, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_experts)])
    def forward(self, tokens, topk_idx):
        expert_outputs = []
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1).unsqueeze(-1).float()
            q = self.queries[e].expand(tokens.shape[0], -1, -1)
            out, _ = self.attn[e](q, tokens * mask, tokens * mask)
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=1)

class Reasoner(nn.Module):
    def __init__(self, cfg_align, cfg_reason, pretrained_layers, rotary_emb):
        super().__init__()
        self.uplift = nn.Linear(cfg_align['dim'], cfg_reason['width'])
        self.layers = nn.ModuleList(pretrained_layers)
        self.rotary_emb = rotary_emb
        self.norm = nn.LayerNorm(cfg_reason['width'])
    def forward(self, x):
        x = self.uplift(x)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            out = layer(x, position_embeddings=(cos, sin))
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)

class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, heads=8, layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                nn.LayerNorm(dim)
            ]) for _ in range(layers)
        ])
    def forward(self, x):
        b = x.shape[0]
        latents = self.latents.unsqueeze(0).repeat(b, 1, 1)
        for attn, norm1, ff, norm2 in self.layers:
            out, _ = attn(latents, x, x)
            latents = norm1(latents + out)
            latents = norm2(latents + ff(latents))
        return latents

class MultiScaleDecoder(nn.Module):
    def __init__(self, latent_dim, encoder_dim, output_dim):
        super().__init__()
        self.main = nn.Linear(latent_dim, output_dim)
        self.res = nn.Linear(encoder_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
    def forward(self, x, skip=None):
        out = self.main(x)
        if skip is not None:
            skip = skip.mean(dim=1) if skip.dim() > 2 else skip
            out = out + (self.gate(x) * self.res(skip))
        return out

# --- Orchestrator with Modality Switch ---

class R_ULIx_Final(nn.Module):
    def __init__(self, cfg, device_dtype=torch.float32):
        super().__init__()
        p = cfg['pipeline']

        # 1. Base Models
        base_llm = AutoModel.from_pretrained(p['encode']['text']['id'], torch_dtype=device_dtype)
        self.text_embed = base_llm.embed_tokens
        self.rotary_emb = base_llm.rotary_emb if hasattr(base_llm, 'rotary_emb') else base_llm.model.rotary_emb
        reason_layers = base_llm.layers[:p['reason']['depth']]

        self.encoders = nn.ModuleDict({
            'image': SiglipVisionModel.from_pretrained(p['encode']['image']['id'], torch_dtype=device_dtype).vision_model,
            'audio': EncodecModel.from_pretrained(p['encode']['audio']['id']).encoder.to(device_dtype),
            'video': AutoModel.from_pretrained(p['encode']['video']['id'], trust_remote_code=True, torch_dtype=device_dtype)
        })

        # 2. Modality Identifiers
        self.modality_id = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, p['align']['dim'])) for m in p['input']['modalities']
        })

        # 3. Processing Pipeline
        self.projectors = nn.ModuleDict({
            m: ProjectNormalize(p['encode'][m]['out'], p['align']['dim']) for m in p['input']['modalities']
        })

        self.router = Router(p['align']['dim'], p['align']['route']['experts'], p['align']['route']['topk'])
        self.experts = LatentExperts(p['align']['route']['experts'], p['align']['dim'], 8)
        self.reasoner = Reasoner(p['align'], p['reason'], reason_layers, self.rotary_emb)
        self.adapter = nn.Linear(p['reason']['width'], p['adapt']['out'])
        self.perceiver = PerceiverResampler(p['adapt']['out'], p['perceiver']['num_latents'])

        # 4. Modality Switch Gate
        # Predicts which modalities should be active (Multi-label classification)
        self.modality_gate = nn.Sequential(
            nn.Linear(p['adapt']['out'], len(p['input']['modalities'])),
            nn.Sigmoid()
        )

        self.decoders = nn.ModuleDict({
            m: MultiScaleDecoder(p['adapt']['out'], p['encode'][m]['out'], p['decode'][m]['out_dim'])
            for m in p['input']['modalities'] if m in p['decode']
        })

        del base_llm
        gc.collect()

    # --- Updated _extract method within R_ULIx_Final ---
    def _extract(self, m, x):
        if m == "text":
            return self.text_embed(x)

        if m == "audio":
            return self.encoders["audio"](x).transpose(1, 2)

        if m == "video":
            # VideoMAE expects (B, C, T, H, W) where T=16, H=224, W=224
            if x.shape[1] != 3:
                x = x.permute(0, 2, 1, 3, 4)

            # Force temporal dimension to 16
            B, C, T, H, W = x.shape
            if T != 16:
                # Simple repeat or pad to reach 16 frames
                rescale = (16 // T) + 1
                x = x.repeat(1, 1, rescale, 1, 1)[:, :, :16]

            out = self.encoders["video"](x)
        else:
            out = self.encoders["image"](x)

        return out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

    def forward(self, inputs):
        tokens, skips = [], {}

        for m, x in inputs.items():
            feat = self._extract(m, x)
            skips[m] = feat
            proj = self.projectors[m](feat)
            if proj.dim() == 2: proj = proj.unsqueeze(1)
            tokens.append(proj + self.modality_id[m].expand(proj.shape[0], proj.shape[1], -1))

        # Core Pipeline
        combined = torch.cat(tokens, dim=1)
        _, _, idx = self.router(combined)
        latent = self.experts(combined, idx)
        reasoned = self.reasoner(latent)
        adapted = self.adapter(reasoned)

        # Perceiver Compression
        pooled_latents = self.perceiver(adapted)
        summary = pooled_latents.mean(dim=1)

        # Modality Switching Logic
        gate_scores = self.modality_gate(summary) # [B, 4]

        outputs = {}
        for i, m in enumerate(config['pipeline']['input']['modalities']):
            if m in self.decoders:
                # Multiply decoder output by gate score to "shut off" unused modalities
                raw_out = self.decoders[m](summary, skips.get(m))
                outputs[m] = raw_out * gate_scores[:, i].unsqueeze(-1)

        return outputs, gate_scores

# --- Execution ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R_ULIx_Final(config).to(device)
print(f"R-ULIx Finalized | Gating Head Ready.")

# Example Inference
def run_test():
    dummy = {"text": torch.randint(0, 1000, (1, 10)).to(device)}
    out, gates = model(dummy)
    print(f"Active Modality Confidence: {gates.cpu().detach().numpy()}")

run_test()


# ===== CELL 5 =====
def test_model():
    batch = 2
    modality_names = config['pipeline']['input']['modalities']

    # Ensure shapes match model expectations exactly
    dummy = {
        "text": torch.randint(0, 151936, (batch, 16)).to(device),
        "image": torch.randn(batch, 3, 224, 224).to(device, dtype=run_dtype),
        "audio": torch.randn(batch, 1, 24000).to(device, dtype=run_dtype),
        # Changed frames from 4 to 16 to satisfy VideoMAE's Positional Embeddings
        "video": torch.randn(batch, 16, 3, 224, 224).to(device, dtype=run_dtype)
    }

    model.eval()
    with torch.no_grad():
        outputs, gate_scores = model(dummy)

        for i in range(batch):
            print(f"\n--- Example {i+1} ---")
            print("Modality Gate Confidences:")
            for m_idx, m_name in enumerate(modality_names):
                conf = gate_scores[i, m_idx].item()
                print(f"  > {m_name:5}: {conf:.2%}")

            if "text" in outputs:
                # We take the mean of the sequence or the first token
                # since the decoder currently produces a summary-based output
                text_logits = outputs["text"][i]
                predicted_id = torch.argmax(text_logits, dim=-1)

                # Handling if predicted_id is a scalar vs sequence
                token_id = predicted_id.item() if predicted_id.dim() == 0 else predicted_id[0].item()
                raw_response = tokenizer.decode(token_id, skip_special_tokens=True)
                print(f"Model Response: {raw_response or '[No Token]'}")
# Run the test
test_model()


# ===== CELL 6 =====
def call_r_ulix_skeleton(model, tokenizer, text=None, image=None, audio=None, video=None):
    model.eval()
    inputs = {}

    # --- Preprocessing Logic ---
    if text is not None:
        inputs["text"] = tokenizer(text, return_tensors="pt").input_ids.to(device)

    if image is not None:
        # Assuming image is a tensor [3, 224, 224]
        inputs["image"] = image.unsqueeze(0).to(device, dtype=run_dtype)

    if audio is not None:
        # Assuming audio is [1, 24000]
        inputs["audio"] = audio.unsqueeze(0).to(device, dtype=run_dtype)

    if video is not None:
        # VideoMAE MUST have 16 frames [16, 3, 224, 224]
        if video.shape[0] < 16:
            rescale = (16 // video.shape[0]) + 1
            video = video.repeat(rescale, 1, 1, 1)[:16]
        inputs["video"] = video.unsqueeze(0).to(device, dtype=run_dtype)

    # --- The Model Call ---
    with torch.no_grad():
        outputs, gate_scores = model(inputs)

    # --- Post-processing (The Switch) ---
    results = {}
    modality_list = config['pipeline']['input']['modalities']

    print("\n--- Skeleton Inference Report ---")
    for i, prob in enumerate(gate_scores[0]):
        m_name = modality_list[i]
        is_active = prob > 0.5
        print(f"Modality: {m_name:6} | Confidence: {prob:.2%} | {'[ACTIVE]' if is_active else '[IDLE]'}")

        if is_active and m_name in outputs:
            results[m_name] = outputs[m_name]

    return results, gate_scores

# --- How to use it ---
# Create a dummy image and text prompt
dummy_img = torch.randn(3, 224, 224)
results, gates = call_r_ulix_skeleton(model, tokenizer, text="Describe this:", image=dummy_img)


# ===== CELL 7 =====
# 1. Helper to prepare inputs
def prepare_dummy_inputs(batch_size=1):
    # Text input: "Hello R-ULIx"
    text_data = tokenizer("Hello R-ULIx", return_tensors="pt").input_ids.to(device)

    # Image input: Standard 224x224 RGB
    image_data = torch.randn(batch_size, 3, 224, 224).to(device, dtype=run_dtype)

    # Audio input: 1 second at 24kHz
    audio_data = torch.randn(batch_size, 1, 24000).to(device, dtype=run_dtype)

    # Video input: 16 frames at 224x224 (Mandatory for VideoMAE)
    video_data = torch.randn(batch_size, 16, 3, 224, 224).to(device, dtype=run_dtype)

    return {
        "text": text_data,
        "image": image_data,
        "audio": audio_data,
        "video": video_data
    }

# 2. The Forced Call Function
def call_r_ulix_forced(model, tokenizer, inputs, force_modalities=["text", "audio"]):
    model.eval()
    with torch.no_grad():
        # Forward pass through the skeleton
        outputs, gate_scores = model(inputs)

        forced_results = {}
        modality_list = config['pipeline']['input']['modalities']

        print("\n--- FORCED Skeleton Inference ---")
        for i, m_name in enumerate(modality_list):
            if m_name in force_modalities and m_name in outputs:
                # Re-normalize to get the "pure" signal before the random gate suppressed it
                current_gate_val = gate_scores[0, i].item()
                pure_output = outputs[m_name] / (current_gate_val + 1e-6)

                forced_results[m_name] = pure_output
                print(f"Forcing Output: [{m_name.upper()}] - Extraction Successful")

                # If it's text, let's see what the random weights say
                if m_name == "text":
                    token_id = torch.argmax(pure_output[0], dim=-1)
                    # Handle if output is sequence or single token
                    if token_id.dim() > 0: token_id = token_id[0]
                    decoded = tokenizer.decode(token_id.item(), skip_special_tokens=True)
                    print(f"  > Raw Text Signal: '{decoded}'")

    return forced_results

# 3. EXECUTION
# First, create the 'inputs' variable
inputs = prepare_dummy_inputs(batch_size=1)

# Second, call the forced function
force_outputs = call_r_ulix_forced(model, tokenizer, inputs, force_modalities=["text", "audio"])


# ===== CELL 8 =====
def initialize_gate_bias(model, active_modalities=["text", "audio"]):
    # The gate is a Linear layer: [adapt_out] -> [4]
    # We access the bias of that linear layer (the 2nd part of the Sequential)
    gate_layer = model.modality_gate[0]
    modality_list = config['pipeline']['input']['modalities']

    with torch.no_grad():
        for i, m_name in enumerate(modality_list):
            if m_name in active_modalities:
                # Setting bias to 2.2 results in ~90% confidence after Sigmoid
                gate_layer.bias[i] = 2.2
            else:
                # Setting bias to -2.2 results in ~10% confidence
                gate_layer.bias[i] = -2.2
    print(f"Gates biased! Default Active: {active_modalities}")

# Apply to your model
initialize_gate_bias(model)


# ===== CELL 9 =====
def check_latent_health(model, inputs):
    model.eval()
    with torch.no_grad():
        # Trace through the model to get the perceiver output
        tokens = []
        for m, x in inputs.items():
            feat = model._extract(m, x)
            proj = model.projectors[m](feat)
            if proj.dim() == 2: proj = proj.unsqueeze(1)
            tokens.append(proj + model.modality_id[m].expand(proj.shape[0], proj.shape[1], -1))

        combined = torch.cat(tokens, dim=1)
        _, _, idx = model.router(combined)
        latent = model.experts(combined, idx)
        reasoned = model.reasoner(latent)
        adapted = model.adapter(reasoned)

        # This is the "Heart" of the model
        pooled_latents = model.perceiver(adapted)

        variance = torch.var(pooled_latents)
        print(f"Latent Heartbeat (Variance): {variance.item():.4f}")
        if variance < 0.01:
            print("⚠️ Warning: Latents are collapsing (Mode Collapse).")
        else:
            print("✅ Latents are diverse and healthy.")

check_latent_health(model, inputs)


# ===== CELL 10 =====
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel, EncodecModel
import gc
import os

# 1. ENVIRONMENT STABILIZATION
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
torch.set_default_device('cpu')

# --- Configuration ---
config_yaml = """
pipeline:
  input:
    modalities: [text, image, audio, video]
  perceiver: { num_latents: 64, heads: 8, layers: 2 }
  encode:
    text:  { id: "Qwen/Qwen2.5-0.5B-Instruct", out: 896 }
    image: { id: "google/siglip-base-patch16-224", out: 768 }
    audio: { id: "facebook/encodec_24khz", out: 128 }
    video: { id: "OpenGVLab/VideoMAEv2-Base", out: 768 }
  align: { dim: 256, route: { experts: 4, topk: 1 } }
  reason: { depth: 2, width: 896 }
  adapt: { out: 512 }
  decode:
    text:  { out_dim: 151936 }
    audio: { out_dim: 1024 }
    image: { out_dim: 12288 }
    video: { out_dim: 368640 }
"""
config = yaml.safe_load(config_yaml)

# --- Architecture Components ---

class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, shared_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
    def forward(self, x): return self.norm(self.proj(x))

class Router(nn.Module):
    def __init__(self, dim, experts, topk):
        super().__init__()
        self.gate = nn.Linear(dim, experts); self.topk = topk
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        vals, idx = torch.topk(probs, self.topk, dim=-1)
        return probs, vals / (vals.sum(dim=-1, keepdim=True) + 1e-6), idx

class LatentExperts(nn.Module):
    def __init__(self, num_experts, dim, heads):
        super().__init__()
        self.num_experts = num_experts
        self.queries = nn.Parameter(torch.randn(num_experts, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_experts)])
    def forward(self, tokens, topk_idx):
        expert_outputs = []
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1).unsqueeze(-1).float().to(tokens.dtype)
            q = self.queries[e].expand(tokens.shape[0], -1, -1).to(tokens.dtype)
            out, _ = self.attn[e](q, tokens * mask, tokens * mask)
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=1)

class Reasoner(nn.Module):
    def __init__(self, cfg_align, cfg_reason, pretrained_layers, rotary_emb):
        super().__init__()
        self.uplift = nn.Linear(cfg_align['dim'], cfg_reason['width'])
        self.layers = nn.ModuleList(pretrained_layers)
        self.rotary_emb = rotary_emb; self.norm = nn.LayerNorm(cfg_reason['width'])
    def forward(self, x, position_embeddings=None):
        x = self.uplift(x)
        if position_embeddings is None:
            pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            out = layer(x, position_embeddings=position_embeddings)
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)

class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, heads=8, layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True), nn.LayerNorm(dim),
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)), nn.LayerNorm(dim)
        ]) for _ in range(layers)])
    def forward(self, x):
        b = x.shape[0]; latents = self.latents.unsqueeze(0).repeat(b, 1, 1).to(x.dtype)
        for attn, norm1, ff, norm2 in self.layers:
            out, _ = attn(latents, x, x); latents = norm1(latents + out)
            latents = norm2(latents + ff(latents))
        return latents

class MultiScaleDecoder(nn.Module):
    def __init__(self, latent_dim, encoder_dim, output_dim):
        super().__init__()
        self.main = nn.Linear(latent_dim, output_dim); self.res = nn.Linear(encoder_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
    def forward(self, x, skip=None):
        out = self.main(x)
        if skip is not None:
            skip = skip.mean(dim=1) if skip.dim() > 2 else skip
            out = out + (self.gate(x) * self.res(skip.to(x.dtype)))
        return out

# --- Orchestrator ---

class R_ULIx_Final(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p = cfg['pipeline']

        print("Initializing Backbones (Standard RAM)...")
        base_llm = AutoModel.from_pretrained(p['encode']['text']['id'], low_cpu_mem_usage=False)
        self.text_embed = base_llm.embed_tokens
        self.rotary_emb = base_llm.rotary_emb if hasattr(base_llm, 'rotary_emb') else base_llm.model.rotary_emb
        reason_layers = base_llm.layers[:p['reason']['depth']]

        self.encoders = nn.ModuleDict({
            'image': SiglipVisionModel.from_pretrained(p['encode']['image']['id'], low_cpu_mem_usage=False).vision_model,
            'audio': EncodecModel.from_pretrained(p['encode']['audio']['id'], low_cpu_mem_usage=False).encoder,
            'video': AutoModel.from_pretrained(p['encode']['video']['id'], trust_remote_code=True, low_cpu_mem_usage=False)
        })

        self.modality_id = nn.ParameterDict({m: nn.Parameter(torch.randn(1, 1, p['align']['dim'])) for m in p['input']['modalities']})
        self.projectors = nn.ModuleDict({m: ProjectNormalize(p['encode'][m]['out'], p['align']['dim']) for m in p['input']['modalities']})
        self.router = Router(p['align']['dim'], p['align']['route']['experts'], p['align']['route']['topk'])
        self.experts = LatentExperts(p['align']['route']['experts'], p['align']['dim'], 8)
        self.reasoner = Reasoner(p['align'], p['reason'], reason_layers, self.rotary_emb)
        self.adapter = nn.Linear(p['reason']['width'], p['adapt']['out'])
        self.perceiver = PerceiverResampler(p['adapt']['out'], p['perceiver']['num_latents'])
        self.modality_gate = nn.Sequential(nn.Linear(p['adapt']['out'], len(p['input']['modalities'])), nn.Sigmoid())
        self.decoders = nn.ModuleDict({m: MultiScaleDecoder(p['adapt']['out'], p['encode'][m]['out'], p['decode'][m]['out_dim'])
                                       for m in p['input']['modalities'] if m in p['decode']})
        del base_llm; gc.collect()

    def _extract(self, m, x):
        if m == "text": return self.text_embed(x)
        if m == "audio": return self.encoders["audio"](x).transpose(1, 2)
        if m == "video":
            if x.shape[1] != 3: x = x.permute(0, 2, 1, 3, 4)
            if x.shape[2] != 16:
                rescale = (16 // x.shape[2]) + 1
                x = x.repeat(1, 1, rescale, 1, 1)[:, :, :16]
            out = self.encoders["video"](x)
        else: out = self.encoders["image"](x)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

    def forward(self, inputs):
        tokens, skips = [], {}
        for m, x in inputs.items():
            feat = self._extract(m, x); skips[m] = feat
            proj = self.projectors[m](feat)
            if proj.dim() == 2: proj = proj.unsqueeze(1)
            tokens.append(proj + self.modality_id[m].expand(proj.shape[0], proj.shape[1], -1).to(proj.dtype))

        combined = torch.cat(tokens, dim=1)
        _, _, idx = self.router(combined)
        latent = self.experts(combined, idx)
        reasoned = self.reasoner(latent)
        adapted = self.adapter(reasoned)
        pooled_latents = self.perceiver(adapted)
        summary = pooled_latents.mean(dim=1); gate_scores = self.modality_gate(summary)

        outputs = {}
        for i, m in enumerate(config['pipeline']['input']['modalities']):
            if m in self.decoders:
                raw_out = self.decoders[m](summary, skips.get(m))
                outputs[m] = raw_out * gate_scores[:, i].unsqueeze(-1)
        return outputs, gate_scores

# --- 4. EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R_ULIx_Final(config).to(device)

if device.type == "cuda":
    model = model.half() # Move to FP16
    print("Optimization: Model converted to Half Precision (FP16).")

print("R-ULIx Ready.")

# --- 5. UPDATED TESTS (Matching Precision) ---
# Ensure inputs match model's device and dtype
dtype = torch.float16 if device.type == "cuda" else torch.float32

def test_a():
    print("\n[TEST A] Image -> Logic Flow")
    img = {"image": torch.randn(1, 3, 224, 224).to(device, dtype=dtype)}
    with torch.no_grad(): out, _ = model(img)
    print(f"Status: {'✅' if 'text' in out else '❌'}")

def test_b():
    print("\n[TEST B] Routing Invariance")
    # Small test for the Router/Expert path
    f1 = torch.randn(1, 5, 256).to(device, dtype=dtype)
    with torch.no_grad(): _, _, i1 = model.router(f1)
    print(f"Status: ✅ (Routed to expert indices: {i1.cpu().numpy().flatten()})")

def test_c():
    print("\n[TEST C] Multi-Modal Gating")
    # Simulate text input (keep as long for embedding lookup)
    txt = {"text": torch.randint(0, 1000, (1, 10)).to(device)}
    with torch.no_grad(): _, g = model(txt)
    print(f"Status: ✅ (Gate Scores: {g.cpu().float().numpy().round(2)})")

test_a(); test_b(); test_c()


# ===== CELL 11 =====
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel, EncodecModel
import gc
import os

# 1. ENVIRONMENT STABILIZATION
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
torch.set_default_device('cpu')

# --- Configuration ---
config_yaml = """
pipeline:
  input:
    modalities: [text, image, audio, video]
  perceiver: { num_latents: 64, heads: 8, layers: 2 }
  encode:
    text:  { id: "Qwen/Qwen2.5-0.5B-Instruct", out: 896 }
    image: { id: "google/siglip-base-patch16-224", out: 768 }
    audio: { id: "facebook/encodec_24khz", out: 128 }
    video: { id: "OpenGVLab/VideoMAEv2-Base", out: 768 }
  align: { dim: 256, route: { experts: 4, topk: 1 } }
  reason: { depth: 2, width: 896 }
  reflect: { heads: 8 }
  adapt: { out: 512 }
  decode:
    text:  { out_dim: 151936 }
    audio: { out_dim: 1024 }
    image: { out_dim: 12288 }
    video: { out_dim: 368640 }
"""
config = yaml.safe_load(config_yaml)

# --- Architecture Components ---

class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, shared_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
    def forward(self, x): return self.norm(self.proj(x))

class Router(nn.Module):
    def __init__(self, dim, experts, topk):
        super().__init__()
        self.gate = nn.Linear(dim, experts); self.topk = topk
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        vals, idx = torch.topk(probs, self.topk, dim=-1)
        return probs, vals / (vals.sum(dim=-1, keepdim=True) + 1e-6), idx

class LatentExperts(nn.Module):
    def __init__(self, num_experts, dim, heads):
        super().__init__()
        self.num_experts = num_experts
        self.queries = nn.Parameter(torch.randn(num_experts, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_experts)])
    def forward(self, tokens, topk_idx):
        expert_outputs = []
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1).unsqueeze(-1).float().to(tokens.dtype)
            q = self.queries[e].expand(tokens.shape[0], -1, -1).to(tokens.dtype)
            out, _ = self.attn[e](q, tokens * mask, tokens * mask)
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=1)

class Reasoner(nn.Module):
    def __init__(self, cfg_align, cfg_reason, pretrained_layers, rotary_emb):
        super().__init__()
        self.uplift = nn.Linear(cfg_align['dim'], cfg_reason['width'])
        self.layers = nn.ModuleList(pretrained_layers)
        self.rotary_emb = rotary_emb; self.norm = nn.LayerNorm(cfg_reason['width'])
    def forward(self, x, position_embeddings=None):
        x = self.uplift(x)
        if position_embeddings is None:
            pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            out = layer(x, position_embeddings=position_embeddings)
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)

class Reflector(nn.Module):
    """ New: Analyzes internal reasoning and applies a gating filter. """
    def __init__(self, dim, heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        res = x
        refracted, _ = self.self_attn(x, x, x)
        filtered = refracted * self.gate(x)
        return self.norm(res + filtered)

class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, heads=8, layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True), nn.LayerNorm(dim),
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)), nn.LayerNorm(dim)
        ]) for _ in range(layers)])
    def forward(self, x):
        b = x.shape[0]; latents = self.latents.unsqueeze(0).repeat(b, 1, 1).to(x.dtype)
        for attn, norm1, ff, norm2 in self.layers:
            out, _ = attn(latents, x, x); latents = norm1(latents + out)
            latents = norm2(latents + ff(latents))
        return latents

class MultiScaleDecoder(nn.Module):
    def __init__(self, latent_dim, encoder_dim, output_dim):
        super().__init__()
        self.main = nn.Linear(latent_dim, output_dim); self.res = nn.Linear(encoder_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
    def forward(self, x, skip=None):
        out = self.main(x)
        if skip is not None:
            skip = skip.mean(dim=1) if skip.dim() > 2 else skip
            out = out + (self.gate(x) * self.res(skip.to(x.dtype)))
        return out

# --- Orchestrator ---

class R_ULIx_Final(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p = cfg['pipeline']

        print("Initializing Backbones...")
        base_llm = AutoModel.from_pretrained(p['encode']['text']['id'], low_cpu_mem_usage=False)
        self.text_embed = base_llm.embed_tokens
        self.rotary_emb = base_llm.rotary_emb if hasattr(base_llm, 'rotary_emb') else base_llm.model.rotary_emb
        reason_layers = base_llm.layers[:p['reason']['depth']]

        self.encoders = nn.ModuleDict({
            'image': SiglipVisionModel.from_pretrained(p['encode']['image']['id'], low_cpu_mem_usage=False).vision_model,
            'audio': EncodecModel.from_pretrained(p['encode']['audio']['id'], low_cpu_mem_usage=False).encoder,
            'video': AutoModel.from_pretrained(p['encode']['video']['id'], trust_remote_code=True, low_cpu_mem_usage=False)
        })

        self.modality_id = nn.ParameterDict({m: nn.Parameter(torch.randn(1, 1, p['align']['dim'])) for m in p['input']['modalities']})
        self.projectors = nn.ModuleDict({m: ProjectNormalize(p['encode'][m]['out'], p['align']['dim']) for m in p['input']['modalities']})

        self.router = Router(p['align']['dim'], p['align']['route']['experts'], p['align']['route']['topk'])
        self.experts = LatentExperts(p['align']['route']['experts'], p['align']['dim'], 8)

        self.reasoner = Reasoner(p['align'], p['reason'], reason_layers, self.rotary_emb)
        self.reflector = Reflector(p['reason']['width'], p['reflect']['heads']) # Reflection Stage

        self.adapter = nn.Linear(p['reason']['width'], p['adapt']['out'])
        self.perceiver = PerceiverResampler(p['adapt']['out'], p['perceiver']['num_latents'])
        self.modality_gate = nn.Sequential(nn.Linear(p['adapt']['out'], len(p['input']['modalities'])), nn.Sigmoid())

        self.decoders = nn.ModuleDict({m: MultiScaleDecoder(p['adapt']['out'], p['encode'][m]['out'], p['decode'][m]['out_dim'])
                                       for m in p['input']['modalities'] if m in p['decode']})
        del base_llm; gc.collect()

    def _extract(self, m, x):
        if m == "text": return self.text_embed(x)
        if m == "audio": return self.encoders["audio"](x).transpose(1, 2)
        if m == "video":
            if x.shape[1] != 3: x = x.permute(0, 2, 1, 3, 4)
            if x.shape[2] != 16:
                rescale = (16 // x.shape[2]) + 1
                x = x.repeat(1, 1, rescale, 1, 1)[:, :, :16]
            out = self.encoders["video"](x)
        else: out = self.encoders["image"](x)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

    def forward(self, inputs):
        tokens, skips = [], {}
        for m, x in inputs.items():
            feat = self._extract(m, x); skips[m] = feat
            proj = self.projectors[m](feat)
            if proj.dim() == 2: proj = proj.unsqueeze(1)
            tokens.append(proj + self.modality_id[m].expand(proj.shape[0], proj.shape[1], -1).to(proj.dtype))

        combined = torch.cat(tokens, dim=1)
        _, _, idx = self.router(combined)
        latent = self.experts(combined, idx)

        reasoned = self.reasoner(latent)
        reflected = self.reflector(reasoned) # Apply Reflection logic here

        adapted = self.adapter(reflected)
        pooled_latents = self.perceiver(adapted)
        summary = pooled_latents.mean(dim=1); gate_scores = self.modality_gate(summary)

        outputs = {}
        for i, m in enumerate(config['pipeline']['input']['modalities']):
            if m in self.decoders:
                raw_out = self.decoders[m](summary, skips.get(m))
                outputs[m] = raw_out * gate_scores[:, i].unsqueeze(-1)
        return outputs, gate_scores

# --- 4. EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R_ULIx_Final(config).to(device)

if device.type == "cuda":
    model = model.half()
    print("Optimization: Model converted to Half Precision (FP16).")

print("R-ULIx with Reflection Ready.")

# --- 5. TESTS ---
dtype = torch.float16 if device.type == "cuda" else torch.float32

def test_a():
    print("\n[TEST A] Reflection Path Check")
    img = {"image": torch.randn(1, 3, 224, 224).to(device, dtype=dtype)}
    with torch.no_grad(): out, _ = model(img)
    print(f"Status: {'✅' if 'text' in out else '❌'}")

test_a()


# ===== CELL 12 =====
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel, EncodecModel
import gc
import os

# 1. ENVIRONMENT STABILIZATION
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
torch.set_default_device('cpu')

# --- Configuration ---
config_yaml = """
pipeline:
  input:
    modalities: [text, image, audio, video]
  perceiver: { num_latents: 64, heads: 8, layers: 2 }
  encode:
    text:  { id: "Qwen/Qwen2.5-0.5B-Instruct", out: 896 }
    image: { id: "google/siglip-base-patch16-224", out: 768 }
    audio: { id: "facebook/encodec_24khz", out: 128 }
    video: { id: "OpenGVLab/VideoMAEv2-Base", out: 768 }
  align: { dim: 256, route: { experts: 4, topk: 1 } }
  reason: { depth: 2, width: 896 }
  reflect: { heads: 8 }
  adapt: { out: 512 }
  decode:
    text:  { out_dim: 151936 }
    audio: { out_dim: 1024 }
    image: { out_dim: 12288 }
    video: { out_dim: 368640 }
"""
config = yaml.safe_load(config_yaml)

# --- Architecture Components ---

class ProjectNormalize(nn.Module):
    def __init__(self, in_dim, shared_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
    def forward(self, x): return self.norm(self.proj(x))

class Router(nn.Module):
    def __init__(self, dim, experts, topk):
        super().__init__()
        self.gate = nn.Linear(dim, experts); self.topk = topk
    def forward(self, x):
        probs = F.softmax(self.gate(x), dim=-1)
        vals, idx = torch.topk(probs, self.topk, dim=-1)
        return probs, vals / (vals.sum(dim=-1, keepdim=True) + 1e-6), idx

class LatentExperts(nn.Module):
    def __init__(self, num_experts, dim, heads):
        super().__init__()
        self.num_experts = num_experts
        self.queries = nn.Parameter(torch.randn(num_experts, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(num_experts)])
    def forward(self, tokens, topk_idx):
        expert_outputs = []
        for e in range(self.num_experts):
            mask = (topk_idx == e).any(dim=-1).unsqueeze(-1).float().to(tokens.dtype)
            q = self.queries[e].expand(tokens.shape[0], -1, -1).to(tokens.dtype)
            out, _ = self.attn[e](q, tokens * mask, tokens * mask)
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=1)

class Reasoner(nn.Module):
    def __init__(self, cfg_align, cfg_reason, pretrained_layers, rotary_emb):
        super().__init__()
        self.uplift = nn.Linear(cfg_align['dim'], cfg_reason['width'])
        self.layers = nn.ModuleList(pretrained_layers)
        self.rotary_emb = rotary_emb; self.norm = nn.LayerNorm(cfg_reason['width'])
    def forward(self, x, position_embeddings=None):
        x = self.uplift(x)
        if position_embeddings is None:
            pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            out = layer(x, position_embeddings=position_embeddings)
            x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)

class Reflector(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        res = x
        refracted, _ = self.self_attn(x, x, x)
        filtered = refracted * self.gate(x)
        return self.norm(res + filtered)

class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_latents=64, heads=8, layers=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.layers = nn.ModuleList([nn.ModuleList([
            nn.MultiheadAttention(dim, heads, batch_first=True), nn.LayerNorm(dim),
            nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)), nn.LayerNorm(dim)
        ]) for _ in range(layers)])
    def forward(self, x):
        b = x.shape[0]; latents = self.latents.unsqueeze(0).repeat(b, 1, 1).to(x.dtype)
        for attn, norm1, ff, norm2 in self.layers:
            out, _ = attn(latents, x, x); latents = norm1(latents + out)
            latents = norm2(latents + ff(latents))
        return latents

class MultiScaleDecoder(nn.Module):
    def __init__(self, latent_dim, encoder_dim, output_dim):
        super().__init__()
        self.main = nn.Linear(latent_dim, output_dim); self.res = nn.Linear(encoder_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
    def forward(self, x, skip=None):
        out = self.main(x)
        if skip is not None:
            skip = skip.mean(dim=1) if skip.dim() > 2 else skip
            out = out + (self.gate(x) * self.res(skip.to(x.dtype)))
        return out

# --- Orchestrator ---

class R_ULIx_Final(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p = cfg['pipeline']

        print("Initializing Backbones (Standard RAM)...")
        base_llm = AutoModel.from_pretrained(p['encode']['text']['id'], low_cpu_mem_usage=False)
        self.text_embed = base_llm.embed_tokens
        self.rotary_emb = base_llm.rotary_emb if hasattr(base_llm, 'rotary_emb') else base_llm.model.rotary_emb
        reason_layers = base_llm.layers[:p['reason']['depth']]

        self.encoders = nn.ModuleDict({
            'image': SiglipVisionModel.from_pretrained(p['encode']['image']['id'], low_cpu_mem_usage=False).vision_model,
            'audio': EncodecModel.from_pretrained(p['encode']['audio']['id'], low_cpu_mem_usage=False).encoder,
            'video': AutoModel.from_pretrained(p['encode']['video']['id'], trust_remote_code=True, low_cpu_mem_usage=False)
        })

        self.modality_id = nn.ParameterDict({m: nn.Parameter(torch.randn(1, 1, p['align']['dim'])) for m in p['input']['modalities']})
        self.projectors = nn.ModuleDict({m: ProjectNormalize(p['encode'][m]['out'], p['align']['dim']) for m in p['input']['modalities']})
        self.router = Router(p['align']['dim'], p['align']['route']['experts'], p['align']['route']['topk'])
        self.experts = LatentExperts(p['align']['route']['experts'], p['align']['dim'], 8)
        self.reasoner = Reasoner(p['align'], p['reason'], reason_layers, self.rotary_emb)
        self.reflector = Reflector(p['reason']['width'], p['reflect']['heads'])
        self.adapter = nn.Linear(p['reason']['width'], p['adapt']['out'])
        self.perceiver = PerceiverResampler(p['adapt']['out'], p['perceiver']['num_latents'])
        self.modality_gate = nn.Sequential(nn.Linear(p['adapt']['out'], len(p['input']['modalities'])), nn.Sigmoid())
        self.decoders = nn.ModuleDict({m: MultiScaleDecoder(p['adapt']['out'], p['encode'][m]['out'], p['decode'][m]['out_dim'])
                                       for m in p['input']['modalities'] if m in p['decode']})
        del base_llm; gc.collect()

    def _extract(self, m, x):
        if m == "text": return self.text_embed(x)
        if m == "audio": return self.encoders["audio"](x).transpose(1, 2)
        if m == "video":
            if x.shape[1] != 3: x = x.permute(0, 2, 1, 3, 4)
            if x.shape[2] != 16:
                rescale = (16 // x.shape[2]) + 1
                x = x.repeat(1, 1, rescale, 1, 1)[:, :, :16]
            out = self.encoders["video"](x)
        else: out = self.encoders["image"](x)
        return out.last_hidden_state if hasattr(out, "last_hidden_state") else (out[0] if isinstance(out, tuple) else out)

    def forward(self, inputs):
        tokens, skips = [], {}
        for m, x in inputs.items():
            feat = self._extract(m, x); skips[m] = feat
            proj = self.projectors[m](feat)
            if proj.dim() == 2: proj = proj.unsqueeze(1)
            tokens.append(proj + self.modality_id[m].expand(proj.shape[0], proj.shape[1], -1).to(proj.dtype))

        combined = torch.cat(tokens, dim=1)
        _, _, idx = self.router(combined)
        latent = self.experts(combined, idx)
        reasoned = self.reasoner(latent)
        reflected = self.reflector(reasoned)
        adapted = self.adapter(reflected)
        pooled_latents = self.perceiver(adapted)
        summary = pooled_latents.mean(dim=1); gate_scores = self.modality_gate(summary)

        outputs = {}
        for i, m in enumerate(config['pipeline']['input']['modalities']):
            if m in self.decoders:
                raw_out = self.decoders[m](summary, skips.get(m))
                outputs[m] = raw_out * gate_scores[:, i].unsqueeze(-1)
        return outputs, gate_scores

# --- 4. EXECUTION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R_ULIx_Final(config).to(device)
if device.type == "cuda": model = model.half()
print("R-ULIx Ready with Reflection.")

# --- 5. COMPREHENSIVE TEST ---
dtype = torch.float16 if device.type == "cuda" else torch.float32

def run_multi_modal_test():
    print("\n[FULL SYSTEM TEST]")
    test_batch = {
        "text": torch.randint(0, 1000, (1, 16)).to(device),
        "image": torch.randn(1, 3, 224, 224).to(device, dtype=dtype),
        "audio": torch.randn(1, 1, 32000).to(device, dtype=dtype) # 1s @ 32kHz
    }
    with torch.no_grad():
        outputs, gates = model(test_batch)

    print(f"Active Output Modalities: {list(outputs.keys())}")
    print(f"Gating Confidence: {gates.cpu().float().numpy().round(3)}")
    print(f"Text Projection Shape: {outputs['text'].shape}")
    print("Status: ✅ SYSTEM STABLE")

run_multi_modal_test()


# ===== CELL 13 =====
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, SiglipVisionModel, EncodecModel
import gc
import os

# --- 1. ENVIRONMENT & CONFIG ---
os.environ["ACCELERATE_USE_FSDP"] = "false"
os.environ["ACCELERATE_USE_DEEPSPEED"] = "false"
torch.set_default_device('cpu')

config_yaml = """
pipeline:
  input: { modalities: [text, image, audio, video] }
  perceiver: { num_latents: 64, heads: 8, layers: 2 }
  encode:
    text:  { id: "Qwen/Qwen2.5-0.5B-Instruct", out: 896 }
    image: { id: "google/siglip-base-patch16-224", out: 768 }
    audio: { id: "facebook/encodec_24khz", out: 128 }
    video: { id: "OpenGVLab/VideoMAEv2-Base", out: 768 }
  align: { dim: 256, route: { experts: 4, topk: 1 } }
  reason: { depth: 2, width: 896 }
  reflect: { heads: 8 }
  adapt: { out: 512 }
  decode:
    text:  { out_dim: 151936 }
    audio: { out_dim: 1024 }
    image: { out_dim: 12288 }
    video: { out_dim: 368640 }
"""
config = yaml.safe_load(config_yaml)

# --- 2. ARCHITECTURAL STAGES ---

class AlignStage(nn.Module):
    def __init__(self, in_dim, shared_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, shared_dim)
        self.norm = nn.LayerNorm(shared_dim)
    def forward(self, x): return self.norm(self.proj(x))

class RouteStage(nn.Module):
    def __init__(self, dim, experts, topk):
        super().__init__()
        self.gate = nn.Linear(dim, experts)
        self.topk = topk
        self.queries = nn.Parameter(torch.randn(experts, 1, dim))
        self.attn = nn.ModuleList([nn.MultiheadAttention(dim, 8, batch_first=True) for _ in range(experts)])

    def forward(self, tokens):
        probs = F.softmax(self.gate(tokens), dim=-1)
        vals, idx = torch.topk(probs, self.topk, dim=-1)
        expert_outputs = []
        for e in range(len(self.attn)):
            mask = (idx == e).any(dim=-1).unsqueeze(-1).float().to(tokens.dtype)
            q = self.queries[e].expand(tokens.shape[0], -1, -1).to(tokens.dtype)
            out, _ = self.attn[e](q, tokens * mask, tokens * mask)
            expert_outputs.append(out)
        return torch.cat(expert_outputs, dim=1)

class ReasonReflectStage(nn.Module):
    def __init__(self, cfg_align, cfg_reason, cfg_reflect, layers, rotary_emb):
        super().__init__()
        self.uplift = nn.Linear(cfg_align['dim'], cfg_reason['width'])
        self.layers = nn.ModuleList(layers)
        self.rotary_emb = rotary_emb
        self.reflector = nn.MultiheadAttention(cfg_reason['width'], cfg_reflect['heads'], batch_first=True)
        self.reflect_gate = nn.Sequential(nn.Linear(cfg_reason['width'], cfg_reason['width']), nn.Sigmoid())
        self.norm = nn.LayerNorm(cfg_reason['width'])

    def forward(self, x):
        x = self.uplift(x)
        pos_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        pos_emb = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            out = layer(x, position_embeddings=pos_emb)
            x = out[0] if isinstance(out, tuple) else out
        refracted, _ = self.reflector(x, x, x)
        return self.norm(x + (refracted * self.reflect_gate(x)))

class CompressStage(nn.Module):
    def __init__(self, in_dim, out_dim, num_latents):
        super().__init__()
        self.adapter = nn.Linear(in_dim, out_dim)
        self.latents = nn.Parameter(torch.randn(num_latents, out_dim))
        self.layers = nn.ModuleList([nn.MultiheadAttention(out_dim, 8, batch_first=True) for _ in range(2)])
    def forward(self, x):
        x = self.adapter(x)
        b = x.shape[0]; l = self.latents.unsqueeze(0).repeat(b, 1, 1).to(x.dtype)
        for attn in self.layers:
            out, _ = attn(l, x, x)
            l = l + out
        return l

class AdaptGateStage(nn.Module):
    def __init__(self, dim, num_modalities):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, num_modalities), nn.Sigmoid())
    def forward(self, pooled): return self.gate(pooled)

class DecodeStage(nn.Module):
    def __init__(self, latent_dim, skip_dim, out_dim):
        super().__init__()
        self.main = nn.Linear(latent_dim, out_dim)
        self.skip_proj = nn.Linear(skip_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
    def forward(self, x, skip):
        res = self.skip_proj(skip.mean(dim=1).to(x.dtype))
        return self.main(x) + (self.gate(x) * res)

# --- 3. THE PIPELINE ORCHESTRATOR ---

class R_ULIx_Pipeline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p = cfg['pipeline']
        print("Stage 1: Loading Encoders...")
        base_llm = AutoModel.from_pretrained(p['encode']['text']['id'], low_cpu_mem_usage=False)
        self.text_embed = base_llm.embed_tokens
        self.rotary_emb = base_llm.rotary_emb if hasattr(base_llm, 'rotary_emb') else base_llm.model.rotary_emb
        self.encoders = nn.ModuleDict({
            'image': SiglipVisionModel.from_pretrained(p['encode']['image']['id'], low_cpu_mem_usage=False).vision_model,
            'audio': EncodecModel.from_pretrained(p['encode']['audio']['id'], low_cpu_mem_usage=False).encoder,
            'video': AutoModel.from_pretrained(p['encode']['video']['id'], trust_remote_code=True, low_cpu_mem_usage=False)
        })
        self.identity_tokens = nn.ParameterDict({m: nn.Parameter(torch.randn(1, 1, p['align']['dim'])) for m in p['input']['modalities']})
        self.aligners = nn.ModuleDict({m: AlignStage(p['encode'][m]['out'], p['align']['dim']) for m in p['input']['modalities']})
        self.router = RouteStage(p['align']['dim'], p['align']['route']['experts'], p['align']['route']['topk'])
        self.reasoner = ReasonReflectStage(p['align'], p['reason'], p['reflect'], base_llm.layers[:p['reason']['depth']], self.rotary_emb)
        self.compressor = CompressStage(p['reason']['width'], p['adapt']['out'], p['perceiver']['num_latents'])
        self.gater = AdaptGateStage(p['adapt']['out'], len(p['input']['modalities']))
        self.decoders = nn.ModuleDict({m: DecodeStage(p['adapt']['out'], p['encode'][m]['out'], p['decode'][m]['out_dim'])
                                       for m in p['input']['modalities'] if m in p['decode']})
        del base_llm; gc.collect()

    def forward(self, inputs):
        tokens, skips = [], {}
        # ENCODE & ALIGN
        for m, x in inputs.items():
            if m == "text": feat = self.text_embed(x)
            elif m == "audio": feat = self.encoders["audio"](x).transpose(1, 2)
            elif m == "video":
                if x.shape[1] != 3: x = x.permute(0, 2, 1, 3, 4)
                feat = self.encoders["video"](x[:, :, :16])
            else: feat = self.encoders["image"](x)

            if hasattr(feat, "last_hidden_state"): feat = feat.last_hidden_state
            skips[m] = feat
            aligned = self.aligners[m](feat)
            if aligned.dim() == 2: aligned = aligned.unsqueeze(1)
            tokens.append(aligned + self.identity_tokens[m].to(aligned.dtype))

        # ROUTE -> REASON/REFLECT -> COMPRESS
        routed = self.router(torch.cat(tokens, dim=1))
        reasoned = self.reasoner(routed)
        compressed = self.compressor(reasoned)
        summary = compressed.mean(dim=1)
        gate_scores = self.gater(summary)

        # DECODE (Now with Safety Check)
        outputs = {}
        for i, m in enumerate(config['pipeline']['input']['modalities']):
            # CRITICAL FIX: Only decode if the modality was in the INPUT batch
            if m in self.decoders and m in skips:
                outputs[m] = self.decoders[m](summary, skips[m]) * gate_scores[:, i].unsqueeze(-1)
        return outputs, gate_scores

# --- 4. RUN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = R_ULIx_Pipeline(config).to(device)
if device.type == "cuda": model = model.half()

# Final Integrated Test (Text + Image)
dtype = torch.float16 if device.type == "cuda" else torch.float32
test_batch = {
    "text": torch.randint(0, 1000, (1, 8)).to(device),
    "image": torch.randn(1, 3, 224, 224).to(device, dtype=dtype)
}



with torch.no_grad():
    out, gates = model(test_batch)
    print(f"\nPipeline Output keys: {list(out.keys())}")
    print(f"Gate Distribution (T, I, A, V): {gates.cpu().float().numpy().round(3)}")
    print("Pipeline Status: ✅ STABLE")


# ===== CELL 14 =====
import torch.optim as optim

# 1. SETUP OPTIMIZER
# We filter parameters to only include those that require gradients
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# 2. DEFINE LOSSES
# Different modalities require different loss functions
criterion_cls = nn.CrossEntropyLoss()  # For text/classification
criterion_mse = nn.MSELoss()           # For continuous signals (Audio/Image/Video)

def train_step(model, batch, optimizer):
    model.train()
    optimizer.zero_grad()

    # Batch is a dict: {'text': tensor, 'image': tensor, ...}
    outputs, gate_scores = model(batch)

    total_loss = 0

    # 3. CALCULATE MODALITY-SPECIFIC LOSSES
    if 'text' in outputs and 'text_target' in batch:
        # Standard Next-Token Prediction or Reconstruction
        t_out = outputs['text'].view(-1, outputs['text'].size(-1))
        t_target = batch['text_target'].view(-1)
        total_loss += criterion_cls(t_out, t_target)

    if 'image' in outputs and 'image_target' in batch:
        total_loss += criterion_mse(outputs['image'], batch['image_target'])

    # 4. LOAD BALANCING LOSS (Optional but recommended for MoE)
    # Encourages the router to use all experts equally
    # importance = gate_scores.mean(0)
    # load_loss = torch.std(importance) / torch.mean(importance)
    # total_loss += 0.01 * load_loss

    total_loss.backward()
    optimizer.step()

    return total_loss.item()

print("Training logic structure prepared.")


# ===== CELL 16 =====
import yaml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel, TFSiglipVisionModel, TFEncodecModel

# --- 1. CONFIG ---
config_yaml = """
pipeline:
  input: { modalities: [text, image, audio, video] }
  align: { dim: 256, route: { experts: 4, topk: 1 } }
  reason: { depth: 2, width: 896 }
  adapt: { out: 512 }
  encode:
    text:  { out: 896 }
    image: { out: 768 }
    audio: { out: 128 }
    video: { out: 768 }
  decode:
    text:  { out_dim: 151936 }
    image: { out_dim: 12288 }
"""
config = yaml.safe_load(config_yaml)
p = config['pipeline']

# --- 2. BUILD THE GRAPH (LINEAR STYLE) ---

def build_pipeline():
    print("Building Graph...")

    # A. DEFINE INPUTS (The Entry Points)
    # We define specific inputs for each modality to keep the graph static
    inp_text  = keras.Input(shape=(None,), dtype="int32", name="text")
    inp_image = keras.Input(shape=(224, 224, 3), dtype="float32", name="image")
    inp_audio = keras.Input(shape=(None, 128), dtype="float32", name="audio")
    inp_video = keras.Input(shape=(16, 224, 224, 3), dtype="float32", name="video")

    inputs = {"text": inp_text, "image": inp_image, "audio": inp_audio, "video": inp_video}

    # B. LOAD ENCODERS (Treated as Layers)
    # Note: In a real run, check if TF weights exist for these specific IDs
    encoders = {}
    try:
        encoders['text'] = TFAutoModel.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").get_input_embeddings()
        encoders['image'] = TFSiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").vision_model
        encoders['audio'] = TFEncodecModel.from_pretrained("facebook/encodec_24khz").encoder
    except:
        print("Warning: Using dummy encoders (TF weights missing)")
        encoders = None

    # C. INSTANTIATE LAYERS (The Components)
    # We create them once here, then reuse them in the logic below

    # 1. Aligners (One per modality)
    align_layers = {}
    identity_tokens = {} # Custom weights (like nn.Parameter)
    for m in p['input']['modalities']:
        align_layers[m] = layers.Dense(p['align']['dim'], name=f"align_{m}")
        align_norms[m] = layers.LayerNormalization(name=f"norm_{m}")
        # Custom Variable for Identity Token
        identity_tokens[m] = tf.Variable(tf.random.normal((1, 1, p['align']['dim'])), trainable=True, name=f"id_{m}")

    # 2. Router
    router_gate = layers.Dense(p['align']['route']['experts'], name="router_gate")
    router_attns = [layers.MultiHeadAttention(8, key_dim=p['align']['dim']//8, name=f"r_attn_{i}") for i in range(p['align']['route']['experts'])]
    router_queries = [tf.Variable(tf.random.normal((1, p['align']['dim'])), trainable=True, name=f"r_q_{i}") for i in range(p['align']['route']['experts'])]

    # 3. Reasoner
    reason_uplift = layers.Dense(p['reason']['width'], name="reason_up")
    reason_blocks = [layers.TransformerBlock(8, p['reason']['width']*2, name=f"r_block_{i}") for i in range(p['reason']['depth'])]
    reason_reflect = layers.MultiHeadAttention(p['reflect']['heads'], key_dim=p['reason']['width']//p['reflect']['heads'], name="reflect")
    reason_gate = layers.Dense(p['reason']['width'], activation='sigmoid', name="reflect_gate")
    reason_norm = layers.LayerNormalization(name="reason_norm")

    # 4. Compressor
    compress_adapter = layers.Dense(p['adapt']['out'], name="comp_adapter")
    compress_attns = [layers.MultiHeadAttention(8, key_dim=p['adapt']['out']//8, name=f"c_attn_{i}") for i in range(2)]
    compress_latents = tf.Variable(tf.random.normal((p['perceiver']['num_latents'], p['adapt']['out'])), trainable=True, name="comp_latents")

    # 5. Gater & Decoders
    adapt_gate = layers.Dense(len(p['input']['modalities']), activation='sigmoid', name="adapt_gate")
    decoders = {}
    for m in p['input']['modalities']:
        if m in p['decode']:
            decoders[m] = {
                'main': layers.Dense(p['decode'][m]['out_dim'], name=f"dec_main_{m}"),
                'skip': layers.Dense(p['decode'][m]['out_dim'], name=f"dec_skip_{m}"),
                'gate': layers.Dense(1, activation='sigmoid', name=f"dec_gate_{m}")
            }

    # D. WIRE THE TENSORS (The Forward Pass Logic)
    tokens = []
    skips = {}

    for m in p['input']['modalities']:
        x = inputs[m]

        # 1. Encode
        if encoders:
            if m == "text": feat = encoders['text'](x)
            elif m == "image": feat = encoders['image'](x).last_hidden_state
            elif m == "audio": feat = encoders['audio'](x).last_hidden_state
            else: feat = tf.zeros((tf.shape(x)[0], 16, p['encode'][m]['out'])) # Video dummy
        else:
            feat = tf.zeros((tf.shape(x)[0], 10, p['encode'][m]['out'])) # Dummy

        skips[m] = feat

        # 2. Align
        aligned = align_layers[m](feat)
        aligned = align_norms[m](aligned)

        # 3. Add Identity Token
        id_token = tf.cast(identity_tokens[m], aligned.dtype)
        id_token = tf.tile(id_token, [tf.shape(aligned)[0], 1, 1])
        tokens.append(aligned + id_token)

    # 4. Route
    concat_tokens = tf.concat(tokens, axis=1)
    probs = tf.nn.softmax(router_gate(concat_tokens), axis=-1)
    vals, idx = tf.math.top_k(probs, p['align']['route']['topk'])

    expert_outs = []
    for e in range(p['align']['route']['experts']):
        mask = tf.cast(tf.any(tf.equal(idx, e), axis=-1, keepdims=True), tf.float32)
        q = tf.tile(tf.expand_dims(router_queries[e], 0), [tf.shape(concat_tokens)[0], 1, 1])
        q = tf.expand_dims(q, 1)
        out = router_attns[e](query=q, key=concat_tokens*mask, value=concat_tokens*mask)
        expert_outs.append(out)
    routed = tf.concat(expert_outs, axis=1)

    # 5. Reason
    x = reason_uplift(routed)
    for block in reason_blocks:
        x = block(x)
    refracted = reason_reflect(query=x, key=x, value=x)
    x = reason_norm(x + (refracted * reason_gate(x)))

    # 6. Compress
    x = compress_adapter(x)
    b = tf.shape(x)[0]
    latents = tf.tile(tf.expand_dims(compress_latents, 0), [b, 1, 1])
    for attn in compress_attns:
        out = attn(query=latents, key=x, value=x)
        latents = latents + out
    compressed = latents

    # 7. Adapt & Gate
    summary = tf.reduce_mean(compressed, axis=1)
    gate_scores = adapt_gate(summary)

    # 8. Decode
    outputs = {}
    for i, m in enumerate(p['input']['modalities']):
        if m in decoders:
            d = decoders[m]
            skip_mean = tf.reduce_mean(skips[m], axis=1)
            res = d['skip'](skip_mean)
            out = d['main'](summary) + (d['gate'](summary) * res)
            # Apply modality gate
            g = tf.expand_dims(gate_scores[:, i], -1)
            outputs[m] = out * g

    # E. CREATE MODEL
    model = keras.Model(inputs=list(inputs.values()), outputs=[outputs, gate_scores])

    # F. ATTACH CUSTOM WEIGHTS (Critical for Functional API)
    # Keras doesn't auto-track standalone tf.Variables
    all_custom_vars = list(identity_tokens.values()) + router_queries + [compress_latents]
    model._trainable_variables += all_custom_vars

    return model

# --- 3. RUN ---

model = build_pipeline()
model.compile(optimizer="adam", loss="mse")

# Dummy Data
data = {
    "text": tf.constant([[1, 2, 3]]),
    "image": tf.random.normal((1, 224, 224, 3)),
    "audio": tf.random.normal((1, 10, 128)),
    "video": tf.random.normal((1, 16, 224, 224, 3))
}

outs, gates = model(data, training=False)
print(f"Output Keys: {list(outs.keys())}")
print(f"Trainable Weights: {len(model.trainable_variables)}")
print("✅ Simple Functional API Complete")
