def process_query_image(img_path,images_features,img_array,Rgb_Colors,VGG_Model): # lut
    windowsize_r = 28
    windowsize_c = 28
    yy = np.array(images_features)
    yy = yy.reshape((len(images_features),4096))
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224))
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_data = img
    new = []
    test_image = gimg
    test_image = test_image.reshape(224,224,)
    # Histogram of Query image
    for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
        for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
            window = test_image[r:r+windowsize_r,c:c+windowsize_c]
            new.append(cv2.calcHist(window,[0,1,2],None,[8,8,8],[0, 256, 0, 256, 0, 256]))    

    
    img_data = np.expand_dims(img_data, axis=0)
    gimg = np.expand_dims(gimg, axis=0)
    
    Rgb = np.append(Rgb_Colors,img_data, axis=0)
    the_images = np.append(img_array, gimg, axis=0)

    img_data = preprocess_input(img_data)

    # Extract Features Query
    new_im_pred = VGG_Model.predict(img_data)
    preds1 = np.append(yy, new_im_pred, axis=0)
    return preds1,new,the_images,Rgb
