'''PROJECT 1: Drug-Drug Interaction (DDI) Prediction Exercise'''
import json
import pickle
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter
import argparse


class Drug:
    '''
    Contains the information and relations for drugs
    '''
    def __init__(self, node_id, morgan_feat, rdkit_feat):
        self.node = node_id
        self.morgan = np.array(morgan_feat)
        self.rdkit = np.array(rdkit_feat)


class DDIClassifier:
    '''
    The classification model
    '''
    def __init__(self, mf_weight = 0.5, rd_weight = 0.5):
        self.mf_weight = mf_weight
        self.rd_weight = rd_weight
        self.train_pairs = [] # to store tuples: drug_a, drug_b, relation_id

    def tanimoto_distance(self, v1, v2): #v1, v2 stand for vector 1 vector 2
        #1-(intersection/union)
        intersection = np.logical_and(v1, v2).sum()
        union = np.logical_or(v1, v2).sum()
        if union !=0:
            return 1-(intersection/union)
        else:
            return 1.0

    def fit(self, training_data, drug_map):
        """
        training_data: List of [id1, id2, relation]
        drug_map: Dictionary containing Drug objects
        """
        for id1, id2, relation in training_data:  # iterating through every known interaction from train.txt
            if id1 in drug_map and id2 in drug_map: # check if molecular features exist for both drugs
                # create a memory for both drugs and their relation
                self.train_pairs.append((drug_map[id1], drug_map[id2], relation))

    def prediction(self, test_drug_a, test_drug_b, k=1):
        """
        test_drug_a, test_drug_b: Objects of the Drug class
        """
        all_distances = []

        for train_a, train_b, relation in self.train_pairs:
            # Calculating Morgan (Tanimoto) distance: how many common structural elements two molecules have relative to their total elements.
            # comparing test_a with train_a and test_b with train_b
            morgan_dist_a= self.tanimoto_distance(test_drug_a.morgan, train_a.morgan)
            morgan_dist_b = self.tanimoto_distance(test_drug_b.morgan, train_b.morgan)
            mf_distance = (morgan_dist_a+morgan_dist_b)/2

            # Calculating RDKit2D Distance (Cosine). cosine: measures the angle of vectors in space: if two drugs have similar physicochemical properties, their vectors point in the same direction.
            # scipy cosine returns 1-similarity
            rdkit_dist_a = cosine(test_drug_a.rdkit, train_a.rdkit)
            rdkit_dist_b = cosine(test_drug_b.rdkit, train_b.rdkit)
            rd_distance = (rdkit_dist_a+rdkit_dist_b)/2

            # Combined distance with 0.5 weights
            total_distance = (self.mf_weight * mf_distance) + (self.rd_weight * rd_distance)

            # Storing the distance along with the relation id
            all_distances.append((total_distance, relation))

        # Sorting from smallest to largest distance
        all_distances.sort(key = lambda x:x[0])

        # k-nearest neighbours
        k_neighbours = [relation for dist, relation in all_distances[:k]]

        # Which relation appears most frequently
        most_common_relation = Counter(k_neighbours).most_common(1)[0][0]

        return most_common_relation

# EXECUTION
def main():
    parser = argparse.ArgumentParser(description='DDI Prediction Tool')
    parser.add_argument('--molecular_feats', required=True)
    parser.add_argument('--relation2id', required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--mode', choices=['train', 'inference'], required=True)
    parser.add_argument('--test')
    parser.add_argument('--cutoff', type=int)
    parser.add_argument('--drugbank_1')
    parser.add_argument('--drugbank_2')

    args = parser.parse_args()



# Loading molecular features and mapping Drug Bank ids

    with open(args.molecular_feats, 'rb') as f:
        data_p = pickle.load(f)
        

    drug_search = {} # mapping dictionary node_id --> Drug_object
    db_id_to_node = {}


    for i in range(len(data_p['Node ID'])):
        node_id = data_p['Node ID'][i]
        db_id = data_p['DrugBank ID'][i]
        drug_obj = Drug(node_id, data_p['Morgan_Features'][i], data_p['RDKit2D_Features'][i])
        drug_search[node_id] = drug_obj
        db_id_to_node[db_id] = node_id

# Loading training data

    training_list = []

    with open(args.train, 'r') as f:
        for line in f:
            parts = list(map(int, line.split())) # splits the line and converts strings to integers
            training_list.append(parts)

# Loading relations JSON file
    with open(args.relation2id, 'r') as f:
        relations_map = json.load(f)

# INITIALIZING THE CLASSIFIER

    classifier = DDIClassifier(mf_weight=0.5, rd_weight=0.5)
    classifier.fit(training_list, drug_search)

    if args.mode == 'train':
        # Reading test file and calculating accuracy
        correct_predictions = 0
        total_tested = 0


        with open(args.test, 'r') as f:
            lines = f.readlines()
            if args.cutoff:
                lines = lines[:args.cutoff] # in case we set a cutoff for speed

            for i, line in enumerate(lines):
                t_a_id, t_b_id, true_relation = map(int, line.split())
                if t_a_id in drug_search and t_b_id in drug_search:
                    predicted_relation = classifier.prediction(drug_search[t_a_id], drug_search[t_b_id], k=1)

                    if predicted_relation == true_relation:
                        correct_predictions+=1

                    total_tested+=1
                    print(f"Sample {i+1}/{len(lines)}: Predicted {predicted_relation}, Real {true_relation}")


        accuracy = correct_predictions/total_tested
        print(f"\nFINAL ACCURACY FOR {total_tested} samples, correct: {correct_predictions},  accuracy: {accuracy:.2f}")

    # Inference Mode - Prediction

    elif args.mode == 'inference':
        id1 = db_id_to_node.get(args.drugbank_1)
        id2 = db_id_to_node.get(args.drugbank_2)

        if id1 is not None and id2 is not None:
            pred_id = classifier.prediction(drug_search[id1], drug_search[id2], k=1)
            # Retrieving the description from JSON
            rel_desc = relations_map.get(str(pred_id), "Unknown Relation")
            print(f"Prediction: {rel_desc}")
        else:
            print("One or both DrugBank IDs not found in molecular features.")

if __name__ == "__main__":
    main()



'''
Inference Mode: The user provides DrugBank IDs (e.g., DB13231).
However, the code works with integer Node IDs (e.g., 0, 1, 2) during training and testing.
The db_id_to_node dictionary handles the translation.
If the user selects inference, the program finds
the molecular features of the two drugs, makes the prediction,
and then searches relation2id.json to print the full
interaction description.
'''

'''
Train Mode (Evaluation)

Evaluate the model on test data:

python implementation.py \
    --molecular_feats DDI_Ben/DDI_Ben/data/initial/drugbank/DB_molecular_feats.pkl \
    --train DDI_Ben/DDI_Ben/data/drugbank_random/train.txt \
    --test DDI_Ben/DDI_Ben/data/drugbank_random/test_S0.txt \
    --cutoff 20 \
    --mode train
'''

'''
Inference Mode (Prediction)

Predict the interaction type between two specific drugs:

python implementation.py \
    --molecular_feats DDI_Ben/DDI_Ben/data/initial/drugbank/DB_molecular_feats.pkl \
    --relation2id DDI_Ben/TextDDI/data/drugbank_random/relation2id.json \
    --train DDI_Ben/DDI_Ben/data/drugbank_random/train.txt \
    --mode inference \
    --drugbank_1 DB13231 \
    --drugbank_2 DB00244
'''
