# CC
Open source Image category classifier

## Usage:

1) create the vocabulary and save it in a file.
          
          ./run.py -v <images_folder> -o <vocab_output_file>
          
        <images_folder>: is a path to a folder containing the images that will construct
                         the vocabulary from them.
        <vocab_output_file>: is a path to a file where the vocabulary will be saved in.
          
2) train the classifier and save it for later use.
          
          ./run.py -t <train_folder> -r <ref_vocab_file> -o <classifier_output_file> -d <categories_dictionary_output_file>
        
        <train_folder>: is a path to a folder where the training images will be found.
                        it should have a sub folder for each category named after its label.
                        Note: if "Cow" and "cow" were 2 labels they will be considered the same.
        <ref_vocab_file>: is a path to a file where the vocabulary saved in step 1.
        <classifier_output_file>: is a path to a file where the trained classifier will be saved in.  
        <categories_dictionary_output_file>: is a path to a file where the categories dictionary will be saved in.
          
3) test and evaluate the performance of the classifier.
          
          ./run.py -e <test_folder> -r <ref_vocab_file> -c <ref_classifier_file> -d <ref_categories_dictionary_file>
          
        <test_folder>: is a path to a folder where the testing images will be found.
                       it should have a sub folder for each category named after its label.
                       this label will be used to determine the correctness of the classifier's prediction.
        <ref_vocab_file>: is a path to a file where the vocabulary saved in step 1.
        <classifier_output_file>: is a path to a file where the trained classifier saved in step 2.  
        <categories_dictionary_output_file>: is a path to a file where the categories dictionary saved in step 2.
        
