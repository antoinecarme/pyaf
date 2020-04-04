# Copyright (C) 2016 Antoine Carme <Antoine.Carme@Laposte.net>
# All rights reserved.

# This file is part of the Python Automatic Forecasting (PyAF) library and is made available under
# the terms of the 3 Clause BSD license

from __future__ import absolute_import

from flask import Flask, jsonify, request #import objects from the Flask model
import os, platform

# from .
import WS_Backend as be;

app = Flask(__name__) #define app using Flask
backends = {};

@app.route("/get_my_ip", methods=["GET"])
def get_my_ip():
    print(request.__dict__)
    return jsonify({'ip': request.remote_addr}), 200

def get_backend():
    common_backend = backends.get(0)
    if(common_backend is None):
        common_backend = be.cFlaskBackend();
        backends[0] = common_backend;
    return common_backend;



# MODELS

def jsonify_models():
    backend = get_backend();
    dict1 = backend.models
    return jsonify({'models' : [{k : v.as_dict()} for k,v in dict1.items()]})    


# GET requests

@app.route('/', methods=['GET'])
def test():
    backend = get_backend();
    return jsonify_models();

@app.route('/models', methods=['GET'])
def returnAllModels():
    backend = get_backend();
    return jsonify_models();



@app.route('/model/<string:name>', methods=['GET'])
def returnOneModel(name):
    backend = get_backend();
    model = backend.get_model(name);
    if(model):
        return jsonify({'model' : model.as_dict()})
    return jsonify({})

@app.route('/model/<string:name>/plot/<string:plot_type>', methods=['GET'])
def returnOneModelPlot(name, plot_type):
    backend = get_backend();
    model = backend.get_model(name);
    model.generatePlots();
    if(model):
        if(plot_type != "all"):
            lPlot_PNG_Base64 = model.mPlots[plot_type];
            lOutput = "<img src=\"data:image/png;base64," + str(lPlot_PNG_Base64) + "\"\>";
        else:
            lOutput = "";
            for k in model.mPlots.keys():
                lPlot_PNG_Base64 = model.mPlots[k];
                lOutput = lOutput + "<img src=\"data:image/png;base64," + str(lPlot_PNG_Base64) + "\"\>\n\n\n";            
        return lOutput;
    return jsonify({})

@app.route('/model/<string:name>/SQL/<string:sql_dialect>', methods=['GET'])
def returnOneModelSQL(name, sql_dialect):
    backend = get_backend();
    model = backend.get_model(name);
    model.generateCode();
    if(model):
        lSQL = model.mSQL[sql_dialect];
        return lSQL
    return jsonify({})

# POST requests

@app.route('/model', methods=['POST'])
def addOneModel():
    backend = get_backend();
    print("JSON : " , request.json);
    print("JSON : " , request.__dict__);
    model = backend.add_model(request.json);
    return jsonify({'model' : model.as_dict()})

# PUT requests 


@app.route('/model/<string:name>', methods=['PUT'])
def editOneModel(name):
    backend = get_backend();
    model = backend.update_model(name , request.json['name']);
    return jsonify({'model' : model.as_dict()})

# DELETE requests

@app.route('/model/<string:name>', methods=['DELETE'])
def removeOneModel(name):
    backend = get_backend();
    backend.remove_model(name);
    return jsonify_models();

if __name__ == '__main__':
    print(platform.platform())
    print(platform.uname())
    print(platform.processor())
    print(platform.python_implementation(), platform.python_version());
    print(os.environ);
    port = int(os.environ.get("PORT", 8081))
    app.run(host='0.0.0.0', port=port, debug=False, processes=3)
