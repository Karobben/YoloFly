import pandas as pd
import numpy as np
import operator

from collections import namedtuple


class head_match():

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    MATCH_result = None
    def area(self, a, b):  # returns None if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy

    def _prev_body_head_vector(self, FLY_matrix, Num_frame, fly):
        """
        Return previous-frame body->head vector (hx-bx, hy-by) for this fly, if available.
        """
        prev_frames = [f for f in FLY_matrix.keys() if f < Num_frame]
        if not prev_frames:
            return None
        prev_frames = sorted(prev_frames, reverse=True)
        for pf in prev_frames:
            rec = FLY_matrix.get(pf, {}).get(fly)
            if not isinstance(rec, dict):
                continue
            body = rec.get("body")
            head = rec.get("head")
            if not isinstance(body, (list, tuple)) or not isinstance(head, (list, tuple)):
                continue
            if len(body) < 2 or len(head) < 2:
                continue
            try:
                bx, by = float(body[0]), float(body[1])
                hx, hy = float(head[0]), float(head[1])
            except Exception:
                continue
            return (hx - bx, hy - by)
        return None

    def _vector_cost_for_pair(self, FLY_matrix, Num_frame, fly, TB_head, head_id, score):
        """
        Cost used for ambiguous assignment:
        - prefer smaller change from previous body->head vector when available
        - fallback to overlap score ordering when previous vector is unavailable
        """
        try:
            body = FLY_matrix[Num_frame][fly]["body"]
            bx = float(body[0])
            by = float(body[1])
            hx = float(TB_head.iloc[int(head_id), 1])
            hy = float(TB_head.iloc[int(head_id), 2])
        except Exception:
            return float("inf")
        prev_vec = self._prev_body_head_vector(FLY_matrix, Num_frame, fly)
        if prev_vec is not None:
            nvx = hx - bx
            nvy = hy - by
            return float(np.hypot(nvx - prev_vec[0], nvy - prev_vec[1]))
        # fallback: lower cost for higher overlap score
        try:
            return float(1.0 - float(score))
        except Exception:
            return float("inf")

    def _overlap_score_for_pair(self, candidates_by_body, fly, head_id):
        hid = int(head_id)
        best = -1.0
        for hh, sc in candidates_by_body.get(fly, []) or []:
            if int(hh) == hid:
                best = max(best, float(sc))
        return best

    def _dedupe_matches_one_head_per_body(self, matches, candidates_by_body, FLY_matrix, Num_frame, TB_head):
        """
        Enforce at most one fly per head row index. If multiple flies map to the same head,
        keep the fly with best overlap to that head, then lower vector cost, then stable fly id.
        """
        if not matches:
            return matches
        by_head = {}
        for fly, hid_s in list(matches.items()):
            hid = int(hid_s)
            by_head.setdefault(hid, []).append(fly)
        for hid, flies in by_head.items():
            if len(flies) <= 1:
                continue

            def rank_key(f):
                ov = self._overlap_score_for_pair(candidates_by_body, f, hid)
                if ov < 0:
                    ov = 0.0
                cost = self._vector_cost_for_pair(FLY_matrix, Num_frame, f, TB_head, hid, ov)
                return (-ov, cost, str(f))

            keeper = min(flies, key=rank_key)
            for f in flies:
                if f != keeper:
                    matches.pop(f, None)
        return matches

    def Sort_uniq(self, MATCH_result):
        '''
        This function is for extract the unique match of the head. Let's say Bod A includes two head a and b, body B has only b. So, we'll give b to B and leave a to A.
        '''
        List = [i.split(":")[0] for i in MATCH_result.keys()]
        Uniq_list = []
        for i in List:
            if List.count(i) ==1:
                Uniq_list += [i]
        Uniq_dic = {}
        for i in Uniq_list:
            for Z in MATCH_result.keys():
                if i in Z:
                    Uniq_dic.update({Z:MATCH_result[Z]})

        for Z in Uniq_dic.keys():
            MATCH_result.pop(Z)
        Uniq_dic.update(MATCH_result)
        MATCH_result = Uniq_dic
        return MATCH_result
        #[i in Z for i,Z in  zip(Uniq_list, MATCH_result.keys())]

    def main(self, FLY_matrix, Num_frame, TB_head, Thread=0.7, strict_overlap_min=0.8):
        """
        1) Build overlap lists: strict map cand_map[fly] = [(head_id, score), ...] with
           score = intersection/head_area >= strict_overlap_min (default 0.8).
           Also build candidates_by_body with Thread (default 0.7) for later resolution/dedupe.
        2) Repeatedly: assign every fly that has exactly one strict candidate; remove that fly's
           key from cand_map; remove that head_id from every other fly's list; repeat until
           no such singleton remains.
        3) Unmatched flies get remaining strict candidates, merged with looser Thread candidates,
           then existing global greedy + per-body greedy + head dedupe.
        """
        TB_head.index = range(len(TB_head.index))
        # Looser overlap table (Thread): used for later greedy assignment + dedupe scoring.
        candidates_by_body = {}
        # Strict overlap map: intersection/head_area >= strict_overlap_min (default 80%).
        cand_map = {fly: [] for fly in FLY_matrix[Num_frame].keys()}
        for fly in FLY_matrix[Num_frame].keys():
            fly_body = FLY_matrix[Num_frame][fly]['body']
            fly_loc = self.Rectangle(
                fly_body[0] - fly_body[2] / 2,
                fly_body[1] - fly_body[3] / 2,
                fly_body[0] + fly_body[2] / 2,
                fly_body[1] + fly_body[3] / 2,
            )
            for ID in range(len(TB_head.index)):
                head_tmp = list(TB_head.iloc[ID, 1:])
                head_loc = self.Rectangle(
                    head_tmp[0] - head_tmp[2] / 2,
                    head_tmp[1] - head_tmp[3] / 2,
                    head_tmp[0] + head_tmp[2] / 2,
                    head_tmp[1] + head_tmp[3] / 2,
                )
                R = self.area(fly_loc, head_loc)
                if R is None:
                    continue
                R = R / (head_tmp[2] * head_tmp[3])
                if R >= Thread:
                    candidates_by_body.setdefault(fly, []).append((ID, R))
                if R >= strict_overlap_min:
                    cand_map[fly].append((ID, R))

        assigned_heads = set()
        matches = {}

        def _strip_head_from_cand_map(hid):
            hid = int(hid)
            for f in list(cand_map.keys()):
                cand_map[f] = [(h, s) for h, s in cand_map[f] if int(h) != hid]
                if not cand_map[f]:
                    del cand_map[f]

        # Round 1: iterative singleton on strict (>=80%) map only.
        # {body_id: [(head_id, score), ...]} — delete matched body key; remove matched head from all values; repeat.
        while True:
            singletons = sorted(
                [fly for fly, lst in cand_map.items() if len(lst) == 1],
                key=str,
            )
            if not singletons:
                break
            progressed = False
            for fly in singletons:
                if fly not in cand_map or len(cand_map[fly]) != 1:
                    continue
                head_id, _score = cand_map[fly][0]
                hid = int(head_id)
                if hid in assigned_heads:
                    del cand_map[fly]
                    progressed = True
                    continue
                matches[fly] = str(hid)
                assigned_heads.add(hid)
                del cand_map[fly]
                _strip_head_from_cand_map(hid)
                progressed = True
            if not progressed:
                break

        # Remaining bodies: strict-map leftovers + any fly not yet matched (empty strict list).
        remaining_candidates = {}
        for fly in FLY_matrix[Num_frame].keys():
            if fly in matches:
                continue
            lst = cand_map.get(fly, [])
            remaining_candidates[fly] = sorted(lst, key=operator.itemgetter(1), reverse=True)

        def _remove_assigned_from_options(head_ids):
            """Remove assigned head row indices from every fly's candidate list (no head twice)."""
            if not head_ids:
                return
            remove_set = set(int(h) for h in head_ids)
            for fly in list(remaining_candidates.keys()):
                if fly in matches:
                    continue
                remaining_candidates[fly] = [
                    (hid, sc) for hid, sc in remaining_candidates[fly]
                    if int(hid) not in remove_set
                ]

        # Merge in looser (Thread) candidates for flies still unresolved, so later phases can use 0.7 overlaps.
        for fly in list(remaining_candidates.keys()):
            if fly in matches:
                continue
            seen = {int(h) for h, _ in remaining_candidates[fly]}
            for hid, sc in candidates_by_body.get(fly, []) or []:
                if int(hid) not in seen:
                    remaining_candidates[fly].append((hid, sc))
                    seen.add(int(hid))
            remaining_candidates[fly] = sorted(
                remaining_candidates[fly], key=operator.itemgetter(1), reverse=True
            )

        body_order = sorted(
            [fly for fly in remaining_candidates if fly not in matches],
            key=lambda fly: (len(remaining_candidates[fly]), str(fly)),
        )

        # Global vector-based assignment for unresolved bodies:
        # - build all (fly, head) candidate pairs
        # - score by vector consistency to previous frame (fallback to overlap score)
        # - sort globally and greedily take best non-conflicting pairs
        unresolved = [fly for fly in body_order if fly not in matches]
        ranked_options = []
        for fly in unresolved:
            cand = remaining_candidates.get(fly, [])
            cand_n = len(cand)
            for head_id, score in cand:
                cost = self._vector_cost_for_pair(
                    FLY_matrix, Num_frame, fly, TB_head, head_id, score
                )
                # Sort priority:
                # 1) vector/score cost (smaller is better)
                # 2) fewer candidates first (more constrained body first)
                # 3) higher overlap score
                # 4) stable deterministic order
                ranked_options.append(
                    (float(cost), int(cand_n), -float(score), str(fly), int(head_id), fly)
                )
        ranked_options.sort()
        for _cost, _cand_n, _neg_score, _fly_key, head_id, fly in ranked_options:
            if fly in matches:
                continue
            hid = int(head_id)
            if hid in assigned_heads:
                continue
            # Skip stale options removed by previous head selections.
            alive_heads = {int(h) for h, _s in remaining_candidates.get(fly, [])}
            if hid not in alive_heads:
                continue
            matches[fly] = str(hid)
            assigned_heads.add(hid)
            _remove_assigned_from_options([hid])

        for fly in body_order:
            if fly in matches:
                continue
            candidates = remaining_candidates[fly]
            for head_id, _score in candidates:
                hid = int(head_id)
                if hid in assigned_heads:
                    continue
                matches[fly] = str(hid)
                assigned_heads.add(hid)
                _remove_assigned_from_options([hid])
                break

        matches = self._dedupe_matches_one_head_per_body(
            matches, candidates_by_body, FLY_matrix, Num_frame, TB_head
        )
        self.MATCH_result = matches
