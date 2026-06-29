#!/usr/bin/env python3
import mlflow
import pandas as pd
from pathlib import Path
import json

mlflow.set_tracking_uri("http://127.0.0.1:5000")

ROOT = Path('.')
RESULT_DIRS = [ROOT / 'results', ROOT / 'LMFlow' / 'results']

metrics_to_log = ['score','accuracy','precision','recall','f1','cohen_kappa','mcc','coverage']

for rd in RESULT_DIRS:
    if not rd.exists():
        continue
    for csv in sorted(rd.glob('*_predictions.csv')):
        base = csv.stem.replace('_predictions','')
        run_name = base
        print(f"Importing {csv} as run {run_name}")
        try:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param('imported_file', str(csv))
                mlflow.log_artifact(str(csv))

                # find matching excel report(s)
                for xlsx in rd.glob(f"{base}*.xlsx"):
                    print(f" Found report: {xlsx}")
                    mlflow.log_artifact(str(xlsx))

                    # try read summary_metrics sheet
                    try:
                        df = pd.read_excel(xlsx, sheet_name='summary_metrics')
                        if not df.empty:
                            row = df.iloc[0]
                            for m in metrics_to_log:
                                if m in row and pd.notna(row[m]):
                                    try:
                                        mlflow.log_metric(m, float(row[m]))
                                    except Exception:
                                        # non-numeric, skip
                                        pass
                                elif m == 'score' and 'accuracy' in row and pd.notna(row['accuracy']):
                                    try:
                                        mlflow.log_metric('score', float(row['accuracy']))
                                    except Exception:
                                        pass
                    except Exception:
                        # no summary_metrics sheet or read error
                        pass

        except Exception as e:
            print(f"Error importing {csv}: {e}")

print("Import complete.")
