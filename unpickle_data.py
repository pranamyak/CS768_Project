# unpickle_data.py
import pickle
import sys

# def unpickle_file(data_dir, name):
#     with open(data_dir + name + '.pkl', "rb") as f:
#         data = pickle.load(f)
#         return data

# unpickle_data.py
import pickle
import sys
import dgl

def unpickle_and_save(data_dir, name):
    with open(data_dir + name + '.pkl', "rb") as f:
        data = pickle.load(f)
        print(len(data))
        train, val, biased_test, unbiased_test = data[0], data[1], data[2], data[3]

        dgl.save_graphs(data_dir + name + "_temp_train_graphs.bin", train.graph_lists)
        dgl.save_graphs(data_dir + name + "_temp_val_graphs.bin", val.graph_lists)
        dgl.save_graphs(data_dir + name + "_temp_biased_test_graphs.bin", biased_test.graph_lists)
        dgl.save_graphs(data_dir + name + "_temp_unbiased_test_graphs.bin", unbiased_test.graph_lists)

        with open(data_dir + name + "_temp_train_labels.pkl", "wb") as f:
            pickle.dump(train.graph_labels, f)

        with open(data_dir + name + "_temp_val_labels.pkl", "wb") as f:
            pickle.dump(val.graph_labels, f)

        with open(data_dir + name + "_temp_biased_test_labels.pkl", "wb") as f:
            pickle.dump(biased_test.graph_labels, f)

        with open(data_dir + name + "_temp_unbiased_test_labels.pkl", "wb") as f:
            pickle.dump(unbiased_test.graph_labels, f)
    
    name1 = "MNIST_unseen_test"
    with open(data_dir + name1 + '.pkl', "rb") as f:
        print("SAVING UNSEEN")
        data = pickle.load(f) 
        unseen_test = data[0]

        dgl.save_graphs(data_dir + name1 + "_temp_unseen_test_graphs.bin", unseen_test.graph_lists)

        with open(data_dir + name1 + "_temp_unseen_test_labels.pkl", "wb") as f:
            pickle.dump(unseen_test.graph_labels, f)
        print("SAVING UNSEEN DONE")


if __name__ == "__main__":
    data_dir = sys.argv[1]
    name = sys.argv[2]
    unpickle_and_save(data_dir, name)
#     for item in data:
#         print(item)