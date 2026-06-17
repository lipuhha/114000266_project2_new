#include <utility>
#include <algorithm>
#include <unordered_map>
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
    int score;
    int depth;
    Move best_move;
    TTFlag flag;
};

static std::unordered_map<uint64_t, TTEntry> g_tt;

/*============================================================
 * Killer Moves Implementation
 *============================================================*/
constexpr int MAX_PLY = 128;
static Move g_killers[2][MAX_PLY];


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
        // Only search captures in quiescence search
        int target_piece = state->piece_at(1 - state->player, action.second.first, action.second.second);
        if(target_piece == 0){
            continue; // Not a capture
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
    auto it = g_tt.find(hash);
    if(it != g_tt.end() && it->second.depth >= depth){
        if(it->second.flag == TT_EXACT){
            return it->second.score;
        } else if(it->second.flag == TT_ALPHA && it->second.score <= alpha){
            return it->second.score;
        } else if(it->second.flag == TT_BETA && it->second.score >= beta){
            return it->second.score;
        }
    }

    history.push(hash);

    if(depth <= 0){
        // Use quiescence search instead of evaluating statically at depth 0
        int score = quiescence(state, alpha, beta, history, ply, ctx, p);
        history.pop(hash);
        return score;
    }

    /* === Alpha-Beta loop with PVS === */
    int best_score = M_MAX;
    bool first = true;
    Move best_move;
    int alpha_orig = alpha;

    // Order moves using TT best move, captures, and killer moves
    bool has_tt = (it != g_tt.end());
    Move tt_best_move = has_tt ? it->second.best_move : Move();

    auto get_move_score = [&](const Move& action) {
        if (has_tt && action == tt_best_move) {
            return 10000;
        }
        bool is_capture = (state->piece_at(1 - state->player, action.second.first, action.second.second) != 0);
        if (is_capture) {
            return 1000;
        }
        if (ply < MAX_PLY) {
            if (action == g_killers[0][ply]) {
                return 100;
            }
            if (action == g_killers[1][ply]) {
                return 90;
            }
        }
        return 0;
    };

    std::stable_sort(state->legal_actions.begin(), state->legal_actions.end(), [&](const Move& a, const Move& b) {
        return get_move_score(a) > get_move_score(b);
    });

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(first){
            // First move: full window
            if(same){
                score = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
            } else {
                score = -eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
            }
            first = false;
        } else {
            // Subsequent moves: zero window search
            if(same){
                score = eval_ctx(next, depth - 1, alpha, alpha + 1, history, ply + 1, ctx, p);
                if(score > alpha && score < beta){
                    // Failed high, re-search with full window
                    score = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
                }
            } else {
                score = -eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, ply + 1, ctx, p);
                if(score > alpha && score < beta){
                    // Failed high, re-search with full window
                    score = -eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
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
            // Store killer move if it's a quiet move
            int target_piece = state->piece_at(1 - state->player, action.second.first, action.second.second);
            if(target_piece == 0 && ply < MAX_PLY){
                if(g_killers[0][ply] != action){
                    g_killers[1][ply] = g_killers[0][ply];
                    g_killers[0][ply] = action;
                }
            }
            break;
        }
    }

    /* === Store in Transposition Table === */
    if(!ctx.stop){
        TTEntry entry;
        entry.score = best_score;
        entry.depth = depth;
        entry.best_move = best_move;
        if(best_score <= alpha_orig){
            entry.flag = TT_ALPHA;
        } else if(best_score >= beta){
            entry.flag = TT_BETA;
        } else {
            entry.flag = TT_EXACT;
        }
        g_tt[hash] = entry;
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

    // Clear Killer Moves table for the new search
    for(int i = 0; i < 2; ++i){
        for(int j = 0; j < MAX_PLY; ++j){
            g_killers[i][j] = Move();
        }
    }

    // Iterative Deepening / TT Move Ordering
    Move tt_best_move;
    bool hash_found = false;
    if(g_prev_hash == state->hash()){
        tt_best_move = g_prev_best_move;
        hash_found = true;
    } else {
        g_tt.clear(); // Clear Transposition Table for a new search/turn
    }

    auto tt_it = g_tt.find(state->hash());
    if(tt_it != g_tt.end()){
        tt_best_move = tt_it->second.best_move;
        hash_found = true;
    }

    // Order moves at the root
    auto get_move_score = [&](const Move& action) {
        if(hash_found && action == tt_best_move){
            return 10000;
        }
        bool is_capture = (state->piece_at(1 - state->player, action.second.first, action.second.second) != 0);
        if(is_capture){
            return 1000;
        }
        return 0;
    };

    std::stable_sort(state->legal_actions.begin(), state->legal_actions.end(), [&](const Move& a, const Move& b) {
        return get_move_score(a) > get_move_score(b);
    });

    bool first = true;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(first){
            // First move: full window
            if(same){
                score = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
            } else {
                score = -eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
            }
            first = false;
        } else {
            // Subsequent moves: zero window search
            if(same){
                score = eval_ctx(next, depth - 1, alpha, alpha + 1, history, 1, ctx, p);
                if(score > alpha && score < beta){
                    // Failed high, re-search with full window
                    score = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
                }
            } else {
                score = -eval_ctx(next, depth - 1, -alpha - 1, -alpha, history, 1, ctx, p);
                if(score > alpha && score < beta){
                    // Failed high, re-search with full window
                    score = -eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
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

    if(!ctx.stop){
        g_prev_best_move = result.best_move;
        g_prev_hash = state->hash();

        // Also store root node results in TT
        TTEntry entry;
        entry.score = best_score;
        entry.depth = depth;
        entry.best_move = result.best_move;
        entry.flag = TT_EXACT;
        g_tt[state->hash()] = entry;
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
    };
}

std::vector<ParamDef> Submission::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
