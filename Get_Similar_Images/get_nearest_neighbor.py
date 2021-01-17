def get_nearest_neighbor_and_similarity(preds1, K):
    dims = 4096
    n_nearest_neighbors = K+1
    # Specify no of trees
    trees = 1000
    file_index_to_file_vector = {}

    # build ann index
    t = AnnoyIndex(dims)
    for i in range(preds1.shape[0]):

        file_vector = preds1[i]
        file_index_to_file_vector[i] = file_vector
        t.add_item(i, file_vector)
    # Build trees
    t.build(trees)

    # Get NN
    for i in range(preds1.shape[0]):
        master_vector = file_index_to_file_vector[i]

        named_nearest_neighbors = []
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

    return nearest_neighbors

