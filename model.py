import data_retrival
import modelling

def main():
    # Execute data retrieval and processing
    data_retrival.get_all_track_data()
    
    # Execute model evaluation
    modelling.evaluate_model()

if __name__ == '__main__':
    main()
