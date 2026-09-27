import model

model.PARAMS["lambda_l2"] = 1.0
model.train(version="v2r_trial", note="v2r_trial min_data_in_leaf=100+lambda_l2=1.0")
