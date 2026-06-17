#include <utility>
#include <algorithm>
#include "state.hpp"
#include "submission.hpp"

static Move g_prev_best_move;
static uint64_t g_prev_hash = 0;

/*============================================================
 * Submission — eval_ctx (Alpha-Beta Pruning)
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
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        ); 
        history.pop(state->hash());
        return score;
    }

    /* === Alpha-Beta loop with PVS === */
    int best_score = M_MAX;
    bool first = true;

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
        }

        // Alpha-Beta pruning logic
        if(best_score > alpha){
            alpha = best_score; // Update alpha (our guaranteed best score)
        }

        if(alpha >= beta){
            // The opponent had a better alternative earlier (beta),
            // so they will avoid this branch entirely. Prune it!
            break;
        }
    }

    history.pop(state->hash());
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

    // Iterative Deepening Move Ordering
    if(g_prev_hash == state->hash()){
        auto it = std::find(state->legal_actions.begin(), state->legal_actions.end(), g_prev_best_move);
        if(it != state->legal_actions.end()){
            std::swap(state->legal_actions[0], *it);
        }
    }

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
    }

    result.score = best_score;
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
