import json, re, os
path = 'jupyter/Prediccion_Rendimiento_Academico_PyCaret-2.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    if cell.get('cell_type') == 'code':
        src = ''.join(cell.get('source', []))
        if any(k in src.lower() for k in ['r2','rmse','lambda','ridge','lasso','elastic','compare_models','holdout','best_models']):
            print(f'\\n--- CELL {i} ---')
            print(src[:4000])
            if 'outputs' in cell and cell['outputs']:
                print('\\nOUTPUTS:')
                for out in cell['outputs']:
                    if out.get('output_type') == 'stream':
                        print(''.join(out.get('text', [])))
                    elif out.get('output_type') == 'execute_result':
                        data = out.get('data', {})
                        if 'text/plain' in data:
                            print(data['text/plain'])
                    elif out.get('output_type') == 'display_data':
                        data = out.get('data', {})
                        for k, v in data.items():
                            if isinstance(v, list):
                                print(''.join(v))
                            elif isinstance(v, str):
                                print(v)
