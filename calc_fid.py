from cleanfid import fid
import sys
if __name__ == '__main__':
    dataset_name = sys.argv[1]
    folder_1 = sys.argv[2]
    cleanfid_args = dict(
                dataset_name=dataset_name,
                dataset_res=64,
                num_gen=10000,
                dataset_split="custom"
                )
    fid_score = fid.compute_fid(folder_1, **cleanfid_args)
    print(f'fid: score: {fid_score}')
    kid_score = fid.compute_kid(folder_1, **cleanfid_args)
    print(f'kid: score: {kid_score}')