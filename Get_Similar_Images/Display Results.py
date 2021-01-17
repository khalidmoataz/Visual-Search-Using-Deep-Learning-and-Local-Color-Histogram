def display_results_after_LCH(the_images,results,nearest_neighbors):
    imgg = the_images[results[0][1]]
    plt.title('Query Image')
    plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))
    fig = plt.figure("Re")
    fig.set_figheight(25)
    fig.set_figwidth(25)
    k=1
    for i in range(1,5):
        imgg = the_images[results[i][1]]
        imgg = imgg.reshape(224,224,3)
        ax = fig.add_subplot(1, 6, k)
        plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        k=k+1

def display_results_before_LCH(the_images,results,nearest_neighbors):
    imgg = the_images[nearest_neighbors[0]]
    plt.title('Query Image')
    plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))
    zfig = plt.figure("Re")
    zfig.set_figheight(25)
    zfig.set_figwidth(25)
    k=1
    for i in range(1,5):
        imgg = the_images[nearest_neighbors[i]]
        imgg = imgg.reshape(224,224,3)
        ax = zfig.add_subplot(1, 6, k)
        plt.imshow(cv2.cvtColor(imgg, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        k=k+1
