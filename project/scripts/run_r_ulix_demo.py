import torch
from project.models.r_ulix import RULIxSkeleton
from project.utils.io import write_text_report
from project.configs.settings import OUTPUTS_DIR

def main():
    model = RULIxSkeleton(input_dim=512, hidden_dim=256, out_dim=64)
    x = torch.randn(2, 16, 512)
    outputs, routes = model(x)
    report = {"output_shape": list(outputs.shape), "route_shape": list(routes.shape)}
    write_text_report(OUTPUTS_DIR / "r_ulix" / "r_ulix_demo.txt", "R-ULIx Demo", [("Summary", str(report))])

if __name__ == "__main__":
    main()
