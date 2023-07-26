from Levenshtein import distance

def print_tensor(x, desc=""):
    if desc:
        print(f"{x.shape} {x.dtype} {x.min()} {x.max()}")
    else:
        print(f"{desc}: {x.shape} {x.dtype} {x.min()} {x.max()}")

def to_phrase(dic, index_lst, validate=False):
    if validate:
        ret = ""
        for val in index_lst:
            val = val.item()
            if val < 3:
                break
            else:
                v = dic[val]
                ret += f"{v}"
        return ret
    else:
        ret = "|"
        for val in index_lst:
            val = val.item()
            if val == 0:
                pass
            else:
                v = dic[val]
                ret += f"{v}|"
        return ret


def calculate_eval(out, gt, iw2, validate=False):
    out = out.cpu()
    gt = gt.cpu()
    ld_val = 0
    word_cnt = 0
    for i in range(out.size(0)):
        gt_phrase = to_phrase(iw2, gt[i], validate)
        out_phrase = to_phrase(iw2, out[i], validate)
        levenshtein_distance = distance(out_phrase, gt_phrase)
        ld_val += levenshtein_distance
        word_cnt += len(gt_phrase)
    eval_val = 1 - (ld_val / word_cnt)
    
    return eval_val