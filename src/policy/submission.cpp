#include <utility>
#include <algorithm>
#include <vector>
#include "state.hpp"
#include "submission.hpp"

static Move g_prev_best_move;
static uint64_t g_prev_hash = 0;

/*============================================================
 * Transposition Table (TT) Implementation
 *============================================================*/
enum TTFlag {
    TT_EXACT = 0,
    TT_ALPHA = 1,
    TT_BETA = 2
};

struct TTEntry {
    uint64_t hash = 0;
    int score = 0;
    int depth = 0;
    Move best_move;
    TTFlag flag = TT_EXACT;
};

static std::vector<TTEntry> g_tt_array;

/*============================================================
 * Killer & History Moves Implementation
 *============================================================*/
constexpr int MAX_PLY = 128;
static Move g_killers[2][MAX_PLY];
static int g_history[7][30];

/*============================================================
 * Helper Functions (TT, Move Ordering, LMR)
 *============================================================*/
static bool tt_probe(uint64_t hash, TTEntry& entry) {
    if (!g_tt_array.empty()) {
        size_t idx = hash % g_tt_array.size();
        if (g_tt_array[idx].hash == hash) {
            entry = g_tt_array[idx];
            return true;
        }
    }
    return false;
}

static void tt_store(uint64_t hash, int depth, int score, Move best_move, TTFlag flag) {
    if (!g_tt_array.empty()) {
        size_t idx = hash % g_tt_array.size();
        if (g_tt_array[idx].hash != hash || depth >= g_tt_array[idx].depth) {
            g_tt_array[idx] = TTEntry{hash, score, depth, best_move, flag};
        }
    }
}

static void order_moves(State* state, std::vector<Move>& moves, const SubParams& p, bool has_tt, const Move& tt_best_move, int ply, bool use_killer) {
    if (!p.use_move_ordering) return;
    auto get_move_score = [&](const Move& action) {
        if (p.use_tt && has_tt && action == tt_best_move) {
            return 10000;
        }
        int victim = state->piece_at(1 - state->player, action.second.first, action.second.second);
        int attacker = state->piece_at(state->player, action.first.first, action.first.second);
        if (victim != 0) {
            return 1000 + PIECE_VALUES[victim] * 10 - PIECE_VALUES[attacker];
        }
        bool is_promotion = (attacker == 1 && (action.second.first == BOARD_H - 1 || action.second.first == 0));
        if (is_promotion) {
            return 900;
        }
        if (use_killer && ply < MAX_PLY) {
            if (action == g_killers[0][ply]) return 100;
            if (action == g_killers[1][ply]) return 90;
        }
        int to_sq = action.second.first * BOARD_W + action.second.second;
        return std::min(80, g_history[attacker][to_sq]);
    };
    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        return get_move_score(a) > get_move_score(b);
    });
}

static int get_lmr_reduction(int depth, int move_count, bool is_capture, bool is_promotion, bool is_killer, const SubParams& p) {
    if (p.use_lmr && depth >= p.lmr_depth_limit && move_count >= p.lmr_full_depth && !is_capture && !is_promotion && !is_killer) {
        if (depth >= p.lmr_depth_limit + 3 && move_count >= p.lmr_full_depth + 5) {
            return 2;
        }
        return 1;
    }
    return 0;
}

/*============================================================
 * Submission — quiescence search
 *============================================================*/
static int quiescence(
    State *state,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const SubParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    if(ctx.nodes % 2048 == 0 && ctx.movetime_ms > 0){
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.search_start).count();
        if(elapsed >= ctx.movetime_ms - 150){
            ctx.stop = true;
            return 0;
        }
    }

    /* === Terminal checks === */
    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    // Standing pat score
    int stand_pat = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
    if (ply >= p.quiescence_max_depth) {
        return stand_pat;
    }
    if(stand_pat >= beta){
        return stand_pat;
    }
    if(stand_pat > alpha){
        alpha = stand_pat;
    }

    // Lazy move generation
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    for(auto& action : state->legal_actions){
        // Only search captures and promotions in quiescence search
        int target_piece = state->piece_at(1 - state->player, action.second.first, action.second.second);
        int moving_piece = state->piece_at(state->player, action.first.first, action.first.second);
        bool is_capture = (target_piece != 0);
        bool is_promotion = (moving_piece == 1 && (action.second.first == BOARD_H - 1 || action.second.first == 0));

        if(!is_capture && !is_promotion){
            continue;
        }

        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(same){
            score = quiescence(next, alpha, beta, history, ply + 1, ctx, p);
        } else {
            score = -quiescence(next, -beta, -alpha, history, ply + 1, ctx, p);
        }

        delete next;

        if(ctx.stop) break;

        if(score >= beta){
            return score; // Beta cutoff
        }
        if(score > alpha){
            alpha = score;
        }
    }

    return alpha;
}

/*============================================================
 * Submission — eval_ctx (Alpha-Beta Pruning with TT & Quiescence)
 *============================================================*/
int Submission::eval_ctx(
    State *state,
    int depth,
    int alpha, // The best score the current player can guarantee
    int beta,  // The best score the opponent can guarantee
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const SubParams& p,
    bool allow_null
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    if(ctx.nodes % 2048 == 0 && ctx.movetime_ms > 0){
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.search_start).count();
        if(elapsed >= ctx.movetime_ms - 150){
            ctx.stop = true;
            return 0;
        }
    }

    /* === Lazy move generation === */
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

    /* === Terminal / leaf checks === */
    if(state->game_state == WIN){
        return P_MAX - ply; // Faster wins are better
    }

    if(state->game_state == DRAW){
        return 0;
    }

    /* === Repetition check === */
    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    /* === Transposition Table Lookup === */
    uint64_t hash = state->hash();
    TTEntry tt_entry;
    bool has_tt = p.use_tt && tt_probe(hash, tt_entry);
    
    if (has_tt && tt_entry.depth >= depth) {
        if(tt_entry.flag == TT_EXACT){
            return tt_entry.score;
        } else if(tt_entry.flag == TT_ALPHA && tt_entry.score <= alpha){
            return tt_entry.score;
        } else if(tt_entry.flag == TT_BETA && tt_entry.score >= beta){
            return tt_entry.score;
        }
    }

    /* === Null Move Pruning === */
    bool has_non_king = false;
    for(int r = 0; r < BOARD_H; r++){
        for(int c = 0; c < BOARD_W; c++){
            int pc = state->piece_at(state->player, r, c);
            if(pc >= 1 && pc <= 5){
                has_non_king = true;
                break;
            }
        }
        if(has_non_king) break;
    }

    if(p.use_null_move && allow_null && depth >= 3 && has_non_king){
        State* null_state = dynamic_cast<State*>(state->create_null_state());
        if(null_state){
            // Avoid null move pruning if we are in check (opponent can capture our king)
            if (null_state->game_state != WIN) {
                int score = -eval_ctx(null_state, depth - 1 - p.null_move_r, -beta, -beta + 1, history, ply + 1, ctx, p, false);
                delete null_state;
                if(score >= beta && !ctx.stop){
                    return score;
                }
            } else {
                delete null_state;
            }
        }
    }

    history.push(hash);

    if(depth <= 0){
        int score;
        if (p.use_quiescence) {
            score = quiescence(state, alpha, beta, history, ply, ctx, p);
        } else {
            score = state->evaluate(p.use_kp_eval, p.use_eval_mobility, &history);
        }
        history.pop(hash);
        return score;
    }

    /* === Alpha-Beta loop with PVS === */
    int best_score = M_MAX;
    bool first = true;
    Move best_move;
    int alpha_orig = alpha;

    // Order moves using TT best move, captures, and killer moves
    Move tt_best_move = has_tt ? tt_entry.best_move : Move();
    order_moves(state, state->legal_actions, p, has_tt, tt_best_move, ply, p.use_killer_moves);

    int move_count = 0;
    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(first){
            // First move: full window
            if(same){
                score = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p, true);
            } else {
                score = -eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p, true);
            }
            first = false;
        } else {
            // Subsequent moves: check LMR
            int victim = state->piece_at(1 - state->player, action.second.first, action.second.second);
            int attacker = state->piece_at(state->player, action.first.first, action.first.second);
            bool is_capture = (victim != 0);
            bool is_promotion = (attacker == 1 && (action.second.first == BOARD_H - 1 || action.second.first == 0));
            bool is_killer = (p.use_killer_moves && ply < MAX_PLY && (action == g_killers[0][ply] || action == g_killers[1][ply]));

            int reduction = get_lmr_reduction(depth, move_count, is_capture, is_promotion, is_killer, p);
            int reduced_depth = std::max(0, depth - 1 - reduction);

            if(same){
                score = eval_ctx(next, reduced_depth, alpha, alpha + 1, history, ply + 1, ctx, p, true);
                if(score > alpha && reduction > 0){
                    score = eval_ctx(next, depth - 1, alpha, alpha + 1, history, ply + 1, ctx, p, true);
                }
                if(score > alpha && score < beta){
                    score = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p, true);
                }
            } else {
                score = -eval_ctx(next, reduced_depth, -alpha - 1, -alpha, history, ply + 1, ctx, p, true);
                if(score > alpha && reduction > 0){
                    score = -eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, ply + 1, ctx, p, true);
                }
                if(score > alpha && score < beta){
                    score = -eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p, true);
                }
            }
        }

        delete next;

        if(ctx.stop) break;

        if(score > best_score){
            best_score = score;
            best_move = action;
        }

        // Alpha-Beta pruning logic
        if(best_score > alpha){
            alpha = best_score; // Update alpha (our guaranteed best score)
        }

        if(alpha >= beta){
            // Prune!
            int target_piece = state->piece_at(1 - state->player, action.second.first, action.second.second);
            if(target_piece == 0){
                int attacker = state->piece_at(state->player, action.first.first, action.first.second);
                int to_sq = action.second.first * BOARD_W + action.second.second;
                g_history[attacker][to_sq] += depth * depth;

                if (p.use_killer_moves && ply < MAX_PLY) {
                    if(g_killers[0][ply] != action){
                        if(g_killers[1][ply] == action){
                            std::swap(g_killers[0][ply], g_killers[1][ply]);
                        } else {
                            g_killers[1][ply] = g_killers[0][ply];
                            g_killers[0][ply] = action;
                        }
                    }
                }
            }
            break;
        }
        move_count++;
    }

    /* === Store in Transposition Table === */
    if(!ctx.stop && p.use_tt){
        TTFlag flag = TT_EXACT;
        if(best_score <= alpha_orig){
            flag = TT_ALPHA;
        } else if(best_score >= beta){
            flag = TT_BETA;
        }
        tt_store(hash, depth, best_score, best_move, flag);
    }

    history.pop(hash);
    return best_score;
}

/*============================================================
 * Submission — search
 *============================================================*/
SearchResult Submission::search(
    State *state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    SubParams p = SubParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }

    int best_score = M_MAX - 10;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    
    // Initial window for the root node
    int alpha = M_MAX;
    int beta = P_MAX;

    // Clear Killer & History tables for the new search
    if (p.use_killer_moves) {
        for(int i = 0; i < 2; ++i){
            for(int j = 0; j < MAX_PLY; ++j){
                g_killers[i][j] = Move();
            }
        }
    }
    for(int i = 0; i < 7; ++i){
        for(int j = 0; j < 30; ++j){
            g_history[i][j] = 0;
        }
    }

    // Handle TT size and initialization
    if (p.use_tt) {
        size_t target_entries = (size_t)p.hash_size_mb * 1024 * 1024 / sizeof(TTEntry);
        if (target_entries < 1024) target_entries = 1024;
        if (g_tt_array.size() != target_entries) {
            g_tt_array.assign(target_entries, TTEntry{});
        }
    } else {
        g_tt_array.clear();
    }

    // Iterative Deepening / TT Move Ordering
    Move tt_best_move;
    bool hash_found = false;
    if (p.use_tt) {
        if(g_prev_hash == state->hash()){
            tt_best_move = g_prev_best_move;
            hash_found = true;
        } else {
            // Clear Transposition Table for a new search/turn
            std::fill(g_tt_array.begin(), g_tt_array.end(), TTEntry{});
        }
        TTEntry tt_entry;
        if (tt_probe(state->hash(), tt_entry)) {
            tt_best_move = tt_entry.best_move;
            hash_found = true;
        }
    } else {
        std::fill(g_tt_array.begin(), g_tt_array.end(), TTEntry{});
    }

    // Order moves at the root
    order_moves(state, state->legal_actions, p, hash_found, tt_best_move, 0, false);

    bool first = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(first){
            // First move: full window
            if(same){
                score = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p, true);
            } else {
                score = -eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p, true);
            }
            first = false;
        } else {
            // Subsequent moves: check LMR
            int victim = state->piece_at(1 - state->player, action.second.first, action.second.second);
            int attacker = state->piece_at(state->player, action.first.first, action.first.second);
            bool is_capture = (victim != 0);
            bool is_promotion = (attacker == 1 && (action.second.first == BOARD_H - 1 || action.second.first == 0));

            int reduction = get_lmr_reduction(depth, move_index, is_capture, is_promotion, false, p);
            int reduced_depth = std::max(0, depth - 1 - reduction);

            if(same){
                score = eval_ctx(next, reduced_depth, alpha, alpha + 1, history, 1, ctx, p, true);
                if(score > alpha && reduction > 0){
                    score = eval_ctx(next, depth - 1, alpha, alpha + 1, history, 1, ctx, p, true);
                }
                if(score > alpha && score < beta){
                    score = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p, true);
                }
            } else {
                score = -eval_ctx(next, reduced_depth, -alpha - 1, -alpha, history, 1, ctx, p, true);
                if(score > alpha && reduction > 0){
                    score = -eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, 1, ctx, p, true);
                }
                if(score > alpha && score < beta){
                    score = -eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p, true);
                }
            }
        }

        delete next;

        if(ctx.stop) break;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
               ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        
        // Update root alpha
        if(best_score > alpha){
            alpha = best_score;
        }

        move_index++;
    }

    if(!ctx.stop && p.use_tt){
        g_prev_best_move = result.best_move;
        g_prev_hash = state->hash();
        tt_store(state->hash(), depth, best_score, result.best_move, TT_EXACT);
    }

    result.score = best_score;
    result.nodes = ctx.nodes; // Fix the diagnostic logs displaying 0 nodes
    if(result.best_move.first != result.best_move.second) {
        result.pv.push_back(result.best_move);
    }
    return result;
}

/*============================================================
 * Submission — default_params / param_defs
 *============================================================*/
ParamMap Submission::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
        {"UseTT", "true"},
        {"UseQuiescence", "true"},
        {"UseKillerMoves", "true"},
        {"UseNullMove", "true"},
        {"UseLMR", "true"},
        {"UseMoveOrdering", "true"},
        {"QuiescenceMaxDepth", "16"},
        {"NullMoveR", "2"},
        {"LMRFullDepth", "3"},
        {"LMRDepthLimit", "3"},
        {"Hash", "64"}
    };
}

std::vector<ParamDef> Submission::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
        {"UseTT", ParamDef::CHECK, "true"},
        {"UseQuiescence", ParamDef::CHECK, "true"},
        {"UseKillerMoves", ParamDef::CHECK, "true"},
        {"UseNullMove", ParamDef::CHECK, "true"},
        {"UseLMR", ParamDef::CHECK, "true"},
        {"UseMoveOrdering", ParamDef::CHECK, "true"},
        {"QuiescenceMaxDepth", ParamDef::SPIN, "16", 0, 64},
        {"NullMoveR", ParamDef::SPIN, "2", 1, 4},
        {"LMRFullDepth", ParamDef::SPIN, "3", 1, 20},
        {"LMRDepthLimit", ParamDef::SPIN, "3", 1, 20},
        {"Hash", ParamDef::SPIN, "64", 1, 1024}
    };
}
