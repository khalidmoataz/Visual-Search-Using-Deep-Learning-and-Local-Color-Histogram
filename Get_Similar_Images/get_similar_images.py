def get_similar_images(nearest_neighbors,images1,query_hist):
    histr = []
    temp = []
    windowsize_r = 28
    windowsize_c = 28
    for i in nearest_neighbors:
        test_image = images1[i]        
        test_image = test_image.reshape(224,224,)
        for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
            for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
                window = test_image[r:r+windowsize_r,c:c+windowsize_c]
                temp.append(cv2.calcHist(window,[0,1,2],None,[8,8,8],[0, 256, 0, 256, 0, 256]))
        histr.append(temp)
        temp = []
    the_sum = 0
    results = {}
    print(len(histr[0]))
    for y in range(0,5):
        the_sum = 0
        for u in range(0,49):
            f = np.array(query_hist[u])
            h = np.array(histr[y][u])

            f = f.flatten()
            h = h.flatten()

            s = dist.euclidean(f,h)
            the_sum =  the_sum + s
        results[nearest_neighbors[y]] = the_sum
    results = sorted([(v, k) for (k, v) in results.items()], reverse = False)
    print(results)
    return results
