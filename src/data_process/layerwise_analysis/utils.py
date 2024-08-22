import numpy as np


def get_segment_idx(
    start_time: float,
    end_time: float,
    len_utt: int,
    stride_sec: float,
    offset: bool,
) -> np.ndarray:
    start_id = int(np.floor(float(start_time) / stride_sec))
    end_id = int(np.ceil(float(end_time) / stride_sec))
    if offset:
        num_offset = int(np.floor((end_id - start_id + 1) / 3))
    else:
        num_offset = 0
    start_id += num_offset
    end_id -= num_offset
    if end_id == start_id:
        end_id += 1
    if end_id == len_utt + 1:
        end_id = len_utt
    # assert (
    #     end_id > start_id
    # ), f"end_id is greater than start_id. {end_id=}, {start_id=}, {len_utt=}."
    if start_id == end_id:
        return np.array([end_id - 1])
    return np.arange(start_id, end_id)


def get_epsilon_lst():
    epsilon_lst = [1e-10, 1e-11, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    epsilon_tuple_lst, epsilon_tuple_lst_rem = [], []
    for i in range(3):
        for j in range(3):
            epsilon_tuple_lst_rem.append((epsilon_lst[i], epsilon_lst[j]))
    np.random.shuffle(epsilon_tuple_lst_rem)
    epsilon_tuple_lst.extend(epsilon_tuple_lst_rem)
    epsilon_tuple_lst_rem = []
    for i in range(3, len(epsilon_lst)):
        for j in range(3, len(epsilon_lst)):
            epsilon_tuple_lst_rem.append((epsilon_lst[i], epsilon_lst[j]))
    np.random.shuffle(epsilon_tuple_lst_rem)
    epsilon_tuple_lst.extend(epsilon_tuple_lst_rem)
    return epsilon_tuple_lst
