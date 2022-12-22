import os
import signal
import torch
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin
from flask_restful import Resource, Api, reqparse
from waitress import serve
from fairseq.models.transformer import TransformerModel
import json
import regex as re
import logging

uescapes = re.compile(r'(?<!\\)\\u[0-9a-fA-F]{4}', re.UNICODE)
ja_char = re.compile(
    r'([\p{IsHan}\p{IsBopo}\p{IsHira}\p{IsKatakana}]+)', re.UNICODE)

last_pre_char = ('」', '』', ')')
first_pre_char = ('『', '「', '(')
post_char = ('「', '」', '”', '“', '"', "'")
remove_char = ('\u200b', '\n')

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

api = Api(app)

def uescape_decode(match):
    return match.group().encode().decode("unicode_escape")

class Api(Resource):
    def post(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('content', type=str, help='content')
            parser.add_argument('message', type=str, help='message')
            args = parser.parse_args()
            message = args['message']
            content = args['content']
            content = uescapes.sub(uescape_decode, content)
            print('\n' + content)
            translate = ja2en.translate(content)
            print(translate)
            # return json.dumps(translate)
            return jsonify(ja2en.translate(content))
        except Exception as e:
            return {'error': str(e)}

    def get(self):
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('content', type=str, help='content')
            args = parser.parse_args()
            content = args['content']
            content = uescapes.sub(uescape_decode, content)
            print('\n' + content)
            translate = ja2en.translate(content)
            print(translate)
            return jsonify(ja2en.translate(content))
            # return jsonify({'content': ja2en.translate(content)})
        except Exception as e:
            return {'error': str(e)}


class Translate:
    def __init__(self):
        self.model = model = TransformerModel.from_pretrained(
            './japaneseModel/',
            checkpoint_file='big.v40.pretrain.pt',
            source_lang="ja",
            target_lang="en",
            bpe='sentencepiece',
            sentencepiece_model='./spmModels/spm.ja.nopretok.model',
            no_repeat_ngram_size=3
        )

        # use pytorch 2.0
        self.ja2en = torch.compile(model, backend="inductor")

        # use cuda instead cpu if available
        if torch.cuda.is_available():
            self.ja2en.cuda()

        # use mps instead cpu if available
        # if torch.backends.mps.is_available():
        #     self.ja2en = ja2en.to(torch.device('mps'))

    def translate(self, data):
        filter_line, isBracket = self.pre_translate_filter(data)
        result = self.ja2en.translate(filter_line)
        result = self.post_translate_filter(result)
        result = self.add_double_quote(result, isBracket)
        return result

    def pre_translate_filter(self, data):
        data = data.strip()
        data = re.sub('([\u200b\n\u3000]+)', '', data)
        # data = re.sub('\u3000', '', data)  # remove "　"
        split = [word for word in data]
        if split[-1] in last_pre_char:
            data = re.sub(r'^.*?(「|『)', '「', data)
            split = [word for word in data]
        if split[0] in first_pre_char and split[-1] in last_pre_char:
            isBracket = True
        else:
            isBracket = False
        return data, isBracket

    def split_text(self, data):
        sArray = re.split('([\.。\?？!！♪:：」』〟]+)', text)
        sArray = [''.join(sArray[i:i+2]) for i in range(0, len(sArray), 2)]
        while('' in sArray):
            sArray.remove('')
        return sArray

    def post_translate_filter(self, data):
        text = data.strip()
        text = re.sub('<unk>', ' ', text)
        text = re.sub('―', '-', text)
        text = re.sub('’', "'", text)
        text = re.sub('`', "'", text)
        text = re.sub('`', "'", text)
        split_text = [word for word in text]
        if split_text[0] in post_char:
            split_text = split_text[1:]
        if split_text[-1] in post_char:
            split_text = split_text[:-1]

        text = ''.join(split_text)
        return text

    def add_double_quote(self, data, isBracket=False):
        en_text = data
        if isBracket:
            en_text = '"' + data + '"'
        return en_text

# warmup
ja2en = Translate()
print('\n' + 'こんにちわ')
print(ja2en.translate('こんにちわ'), '\n')

api.add_resource(Api, '/')

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=14366)



