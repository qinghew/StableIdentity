import torch
import numpy as np
from typing import List


def get_rep_pos(tokenized: torch.Tensor, rep_tokens: list):
    pos_list = []
    # for token in rep_tokens:
    #     pos_list.append(torch.where(tokenized == token).cpu().numpy())
    for token in rep_tokens:
        pos_list = torch.stack(torch.where(tokenized == token)).T.tolist()
    return pos_list


def shift_tensor_dim0(ori: torch.Tensor, r_pos: List[np.ndarray], reps: int):
    assert reps >= 1
    device = ori.device
    d = ori.shape[0]
    offset = np.zeros(d, dtype=np.int64)
    r_pos_cat = np.concatenate(r_pos)
    for p in r_pos_cat:
        offset[p + 1:] += (reps - 1)

    ''' shift words '''
    r_cnt = r_pos_cat.shape[0]  # placeholders occurring times before replacement
    target_pos = (np.arange(d) + offset)[:d - r_cnt * (reps - 1)]  # drop the last
    ori[target_pos] = ori[np.arange(target_pos.shape[0])]
    # print('target_pos:', target_pos)

    ''' fill blanks with repeat words '''
    rep_final_pos: np.ndarray = target_pos[r_pos_cat].repeat(reps) + np.tile(np.arange(reps), r_cnt)
    ori[rep_final_pos] = ori[target_pos[r_pos_cat].repeat(reps)]

    ''' return repeat final position list '''
    rep_final_pos_list = []
    lo = 0
    for i in range(len(r_pos)):
        r_one_times = r_pos[i].shape[0]  # 'sks' occurring times before replacement
        r_one_nums = r_one_times * reps  # 'sks' total times after replacement
        rep_final_pos_list.append(rep_final_pos[lo: lo + r_one_nums].reshape(r_one_times, reps))
        lo += r_one_nums
    # print('rep_final_pos_list:', rep_final_pos_list)
    return ori, rep_final_pos_list


def _test_get_rep_pos():
    tokenized = torch.LongTensor([0, 1, 2, 2, 3, 4, 5, 6, 7, 99] + [99] * 20)
    print('[from]:', tokenized)
    rep_tokens = [2, 6]
    rep_times = 2

    rep_pos = get_rep_pos(tokenized, rep_tokens)
    print('[rep_pos]:', rep_pos)
    res, rep_pos_final = shift_tensor_dim0(tokenized, rep_pos, rep_times)
    print('[to]:', res)
    print('[final pos]:', rep_pos_final)


def _test_shift_tensor_dim0():
    embedded = torch.arange(20)
    print(embedded)
    pos = np.array([3, 6, 8])
    times = 1
    output = shift_tensor_dim0(embedded, pos, times)
    print(output)


if __name__ == "__main__":
    _test_get_rep_pos()
