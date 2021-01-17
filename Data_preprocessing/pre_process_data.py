VGG_Model = load_model('vgg_4096.h5',compile=False)

def pre_process_data(the_path,model):
    images_features = []
    img_array = []
    Rgb_Colors = []
    for path, _, files in os.walk(the_path):
        for file in files:
            img_path = path + str('\\') + file
            img = cv2.imread(img_path)
            img = cv2.resize(img,(224,224))
            gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_data = img
            img_array.append(gimg)
            Rgb_Colors.append(img_data)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)

            images_features.append(model.predict(img_data))

    return img_array,images_features,Rgb_Colors 
