
import random
import pandas as pd
import torch
from werkzeug.utils import secure_filename
from wtforms.validators import DataRequired, Length, Email, EqualTo, NumberRange, regexp
from wtforms.fields import (StringField, PasswordField, DateField, BooleanField,
                            SelectField, SelectMultipleField, TextAreaField,
                            RadioField, IntegerField, DecimalField, SubmitField, FileField)
from wtforms import Form, validators
from flask import Flask, render_template, request, redirect, url_for
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from flask_bootstrap import Bootstrap
import os
from  pandarallel import pandarallel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__), '../..')))
import yaml
from models.parrot_model import ParrotConditionPredictionModel

from models.utils import caonicalize_rxn_smiles, inference_load, get_output_results

model_work_path = os.path.abspath(os.path.join(os.path.abspath(__file__), '../..'))

rxn_fig_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static', 'rxn_fig'))
if not os.path.exists(rxn_fig_path):
    os.makedirs(rxn_fig_path)

app = Flask(__name__)
WTF_CSRF_ENABLED = True  # prevents CSRF attacks



def svg2file(fname, svg_text):
    with open(fname, 'w') as f:
        f.write(svg_text)
        
def reaction2svg(Reaction, path):
    # smi = ''.join(smi.split(' '))
    # mol = Chem.MolFromSmiles(smi)
    d = Draw.MolDraw2DSVG(450, 450)
    d.DrawReaction(Reaction)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', '').replace('y=\'0.0\'>', 'y=\'0.0\' fill=\'rgb(255,255,255,0)\'>')  # 使用replace将原始白色的svg背景变透明
    # svg2 = svg.replace('svg:', '').replace(';fill:#FFFFFF;stroke:none', 'rgb(255,255,255,0)')
    svg2 = svg.replace('svg:', '')
    svg2file(path, svg2)
    return '\n'.join(svg2.split('\n')[8:-1])

class ParrotInferenceAPI:
    def __init__(self) -> None:
        config = yaml.load(open('../configs/config_inference_use_uspto.yaml', "r"),
                       Loader=yaml.FullLoader)
        

        
        
        print(
        '\n########################\nInference configs:\n########################\n'
        )
        print(yaml.dump(config))
        print('########################\n')

        
        
        model_args = config['model_args']
        model_args['use_multiprocessing'] = False
        model_args['best_model_dir'] = os.path.abspath(os.path.join(model_work_path, model_args['best_model_dir']))
        model_args['output_dir'] = os.path.abspath(os.path.join(model_work_path, model_args['output_dir']))
        model_args['pretrained_path'] = os.path.abspath(os.path.join(model_work_path, model_args['pretrained_path']))
        
        
        dataset_args = config['dataset_args']
        dataset_args['dataset_root'] = os.path.abspath(os.path.join(model_work_path, dataset_args['dataset_root']))
        
        
        inference_args = config['inference_args']
        try:
            model_args['use_temperature'] = dataset_args['use_temperature']
            print('Using Temperature:', model_args['use_temperature'])
        except:
            print('No temperature information is specified!')
        condition_label_mapping = inference_load(**dataset_args)
        model_args['decoder_args'].update({
            'tgt_vocab_size':
            len(condition_label_mapping[0]),
            'condition_label_mapping':
            condition_label_mapping
        })

        trained_path = model_args['best_model_dir']
        model = ParrotConditionPredictionModel(
            "bert",
            trained_path,
            args=model_args,
            use_cuda=True if torch.cuda.is_available() else False,
            cuda_device=0 
            )
        self.model = model
        self.config = config
        self.model_args = model_args
        self.dataset_args = dataset_args

        self.inference_args = inference_args
        
        
    def predict(self, input_rxn_smiles):

        # pandarallel.initialize(nb_workers=10, progress_bar=True)
        test_df = pd.DataFrame({
            'text': input_rxn_smiles,
            'labels': [[0] * 7] * len(input_rxn_smiles) if not self.model_args['use_temperature'] else [[0] * 8] * len(input_rxn_smiles) 
        })
        print('Caonicalize reaction smiles and remove invalid reaction...')
        test_df['text'] = test_df.text.apply(
            lambda x: caonicalize_rxn_smiles(x))
        test_df = test_df.loc[test_df['text'] != ''].reset_index(drop=True)
        beam = self.inference_args['beam']
        pred_conditions, pred_temperatures = self.model.condition_beam_search(
            test_df,
            output_dir=self.model_args['best_model_dir'],
            beam=beam,
            test_batch_size=8,
            calculate_topk_accuracy=False
        )
        output_results = get_output_results(test_df.text.tolist(), pred_conditions,
                              pred_temperatures, output_dataframe=False)
        print('Done!')
        return output_results
        
        
        

@app.before_first_request
def first_request():
    app.Parrot = ParrotInferenceAPI()
    return app.Parrot



class RXNForm(Form):
    drawn_smiles = StringField(label='drawn_smiles')
    smiles = TextAreaField(label='smiles')
    file = FileField(label=None, render_kw={'class': 'form-control'})
    submit = SubmitField('Submit')



@app.route('/')
def index():
    form = RXNForm(request.form)
    return render_template('main/index.html', form=form)

@app.route('/results', methods=['GET', 'POST'])
def results():
    
    all_list = []
    sdf_path = None
    if request.method == 'POST':

        form = RXNForm(request.form, request.files)

        print(form.drawn_smiles.data, form.file.data,
              form.smiles.data)

        if form.drawn_smiles.data:
            all_list.append(form.drawn_smiles.data)

        import re

        if form.smiles.data:
            smiles = re.split(': |, |:|,| |。|，|；|；|\r\n', form.smiles.data)
            print(smiles)
            all_list.extend(smiles)

        if request.files['file']:

            file = request.files['file']
            filename = secure_filename(file.filename)
            if request.files.get('file').filename.split('.')[-1] == 'txt':
                for line in request.files.get('file'):
                    print(line.decode('utf-8').strip())
                    all_list.append(line.decode('utf-8').strip())
        all_list = list(set(all_list))
        print(all_list)
    else:
        # form = MolForm(request.form)
        # return render_template('works/works_gostar_results.html', form=form)
        return redirect(url_for('index'))
    
    try:
        output_results = app.Parrot.predict(all_list)
        assert isinstance(output_results, list)
        output_df = pd.concat(output_results, axis=0)
        print(output_df)
        table_name =  '{}.csv'.format(random.randint(0, 1000))
        output_df.to_csv(os.path.join('./static/table', table_name), index=False)
        reactions = [AllChem.ReactionFromSmarts(x, useSmiles=True) for x in output_df['rxn_smiles'].tolist() if x]
        rxn_fig_names = [f'{random.randint(0, 1000)}.svg' for _ in range(len(reactions))]
        rxn_fig_path_list = [os.path.join(rxn_fig_path, name) for name in rxn_fig_names]
        [reaction2svg(rxn, name) for rxn, name in zip(reactions, rxn_fig_path_list)]
        reaction_p_results = int(len(output_df) / len(reactions))
        title = output_df.columns.tolist()
        title[0] = 'Reactions'
        
        output_to_show = [[df.T[col].tolist()[1:] for col in df.T.columns.tolist()] for df in output_results]
        
        
        return render_template('main/results.html', ret={
            'error': None,
            'title': title,
            'output':output_to_show,
            'rxn_fig_names': rxn_fig_names,
            'rowspan': reaction_p_results,
            'table_url': table_name
        })
        
    except Exception as e:
        print(e)
        return render_template("main/results.html", ret={
            # 'form': form,
            'error': 'Input is not valid!',
            'output': [],
            # 'csv_id': csv_id,
        })
        

if __name__ == '__main__':
    from pathlib import Path
    cur_file_path = Path(__file__).resolve().parent  # Path.cwd().parent   #
    app.config['UPLOAD_FOLDER'] = cur_file_path/'upload'
    app.config['MAX_CONTENT_PATH'] = 2**10
    Bootstrap(app)
    app.run(host='0.0.0.0', port=8000, debug=True)
