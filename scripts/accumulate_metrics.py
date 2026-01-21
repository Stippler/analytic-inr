import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import json
from itertools import product

root_dir = Path(__file__).parent.parent

# constants for latex table highlighting

N_TOP_HIGHLIGHTED = 3
HIGHLIGHT_SHADE_FROM, HIGHLIGHT_SHADE_TO = 15, 50
HIGHLIGHT_SHADE_RANGE = HIGHLIGHT_SHADE_TO - HIGHLIGHT_SHADE_FROM
HIGHLIGHT_SHADE_STEPSIZE = (HIGHLIGHT_SHADE_RANGE // (N_TOP_HIGHLIGHTED - 1))
HIGHLIGHT_STYLES = [
    f"cellcolor:{{tab_color!{HIGHLIGHT_SHADE_FROM + i * HIGHLIGHT_SHADE_STEPSIZE}}}"
    for i in range(N_TOP_HIGHLIGHTED)
]

COLUMNS = ["arch", "mesh", "chamfer_l2", "hausdorff", "precision", "recall", "f_score"]

def highlight_top_n_cells(styler, df, styles, ascending=False):
    def make_style_func(column):
        sorted_vals = df.sort_values(column, ascending=ascending)[column].values

        def style_func(v):
            for i in range(min(len(sorted_vals), len(styles))):
                if (not ascending and v >= sorted_vals[i]) or (ascending and v <= sorted_vals[i]):
                    return styles[i]
            return ""

        return style_func

    for column in df.columns:
        styler = styler.map(make_style_func(column), subset=column)
    return styler

def print_latex_table(results, ascending=False):
    df = pd.DataFrame(results, columns=COLUMNS)
    df = df.groupby('arch').mean(numeric_only=True).reset_index()
    
    # Select only the metrics we care about
    df.columns = [col.replace("_", r"\_") for col in df.columns]
    cols = [c for c in df.columns if c in ["chamfer\\_l2", "hausdorff"]]
    df["arch"] = df["arch"].str.replace("_", r"\_", regex=False)
    

    # Multiply them by 1000
    df[cols] = df[cols] * 1000
    
    df = df.round(3)
    df = df.set_index("arch")

    styler = df.style.pipe(
        highlight_top_n_cells,
        df=df,
        styles=HIGHLIGHT_STYLES[::-1],
        ascending=ascending,
    )
    styler = styler.format(precision=3)

    tex_path = "output.tex"
    styler.to_latex(tex_path, hrules=True)
    print(styler.to_latex(hrules=True))
    
def create_metrics_df(all_calls):
    results = []
    for (arch, mesh) in all_calls:
        try:
            net_dir = root_dir / "nets" / args.task / arch / Path(mesh).stem
            if not net_dir.exists():
                continue
            
            metrics_path = net_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            results.append(
                [arch, mesh] + list([m for m in metrics.values() if type(m) == float])
            )

        except Exception as e:
            print(e)
            continue
        
    return results
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Trains SDF neural networks from point clouds.")
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    
    calls = []
    # for x in Path("outputs/Stanford_armadillo").iterdir():
    #     if x.is_dir() and 'quantile' in x.name:
    #         with open((x / 'Stanford_armadillo' / 'metrics.json'), 'r') as f:
    #             metrics = json.load(f)

    #         calls.append([x.name, "Stanford_armadillo", metrics['chamfer_l2'], metrics['hausdorff'], metrics['precision'], metrics['recall'], metrics['f_score']])
    # # results = create_metrics_df(calls)
    # print_latex_table(calls, ascending=True)
    
    task_dir = root_dir / "nets" / args.task
    mesh_dir = root_dir / "data" / "meshes"
    
    arch = (p.name for p in task_dir.iterdir())
    mesh = (p.name for p in mesh_dir.iterdir())
    
    all_calls = list(product(arch, mesh))
    results = create_metrics_df(all_calls)
    print_latex_table(results, ascending=True)
