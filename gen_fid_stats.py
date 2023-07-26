from cleanfid import fid
import sys
if __name__ == '__main__':
    custom_name = sys.argv[1]
    dataset_path = sys.argv[2]
    # TODO: remove saved stats if you want to reproduce.
    # fid.remove_custom_stats(custom_name, mode="clean")
    print(f'Generating clean-fid for dataset {custom_name} located at {dataset_path}')
    fid.make_custom_stats(custom_name, dataset_path, mode="clean")