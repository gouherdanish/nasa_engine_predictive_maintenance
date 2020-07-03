import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from flask import Flask, jsonify, request, abort, make_response
#from sklearn.linear_model import LinearRegression
#from sklearn.feature_selection import VarianceThreshold
import pickle

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/')
def index():
    return {'hello': 'world'}

@app.route('/engine_hist/', methods=["GET"])
def get_engines_hist_data():
    cycles_list = eng_df['cycles_to_fail'].tolist()
    return jsonify({"result":cycles_list})

@app.route('/features/', methods=["GET"])
def select_features():
    sub_dict_list = [train_df_std.loc[i].to_dict() for i in range(len(train_df_std))]
    return jsonify({"result":sub_dict_list})

@app.route('/engine_scatter/', methods=["POST"])
def get_engines_scatter_data():
    if not request.json:
        abort(400)
    # get data
    req = request.get_json(force=True)
    print(req)
    engine_id = int(req["engine_id"])
    sensor_id = req["sensor_id"].lower()
    
    sub_df = train_df.loc[train_df.id == engine_id, ['cycle',sensor_id]].reset_index(drop=True)
    sub_df.columns = ['cycle','sensor_reading']
#    sub_df = train_df[train_df.id == engine_id][['cycle',sensor_id]]
    sub_dict_list = [sub_df.loc[i].to_dict() for i in range(len(sub_df))]
    return jsonify({"result":sub_dict_list})

@app.route('/engine/', methods=["POST"])
def get_engines_data():
    if not request.json:
        abort(400)
    req = request.json
    engine_type = req["engine_type"].lower()
    num_of_engines = int(req["num_of_engines"])
    
    if engine_type == "best":
        sub_df = eng_df[:num_of_engines]
    elif engine_type == "worst":
        sub_df = eng_df[-num_of_engines:].reset_index(drop=True)
    else:
        abort(400)
        
    sub_dict_list = [sub_df.loc[i].to_dict() for i in range(num_of_engines)]
        
    return jsonify({"result": sub_dict_list})

# =============================================================================
# @app.route('/fit_model/', methods=["POST"])
# def create_model():
#     if not request.json:
#         abort(400)
#     req = request.json
#     model = req["model"]
#     model_type = req["model_type"]
#     engine = req["engine"]
#     sensor = req["sensor"]
# =============================================================================
    
@app.route('/predict/', methods=["POST"])
def predict_engine_life():
    if not request.json:
        abort(400)
    
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    print(data)

    # predictions
    result = model.predict(poly_feat.fit_transform(data_df))
    
    # Checking if predicted engine life is negative
    res = int(result[0]) if int(result[0]) > 0 else 0

    # return data
    return jsonify({"remaining_useful_life": res})

if __name__ == '__main__':
    
    # Column names for input data
    col_names = ['id','cycle','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23']
    
    # Reading Data
    train_df = pd.read_csv("PM_train.txt",sep=' ',header = None,names = col_names).dropna(axis=1)
    
    # Engine ID and correspoding number of Flight Cycles undergone till it fails
    eng_df = train_df.id.value_counts().rename_axis('id').reset_index(name='cycles_to_fail')
    
    # Excluding Engine id and cycle number from the input features
    features = ['setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
    
    # Finding the standard deviation of each feature and storing in a dataframe in descending order of std
    train_df_std = train_df[features].std().sort_values(ascending=False).rename_axis('feature').reset_index(name='std')
    
    # load model
    model = pickle.load(open('mvar_polyreg_model.pkl','rb'))
    poly_feat = PolynomialFeatures(degree=2)
    print("loaded OK")
    app.run(debug=True)