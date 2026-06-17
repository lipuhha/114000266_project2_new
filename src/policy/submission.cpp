#include <utility>
#include <algorithm>
#include "state.hpp"
#include "submission.hpp"


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

    /* === Alpha-Beta loop === */
    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(same){
            // If the next turn is ours (e.g. multi-step moves, not standard in minichess but supported), 
            // the perspective is the same, so we pass alpha and beta directly.
            score = eval_ctx(next, depth - 1, alpha, beta, history, ply + 1, ctx, p);
        } else {
            // Usually, the next turn belongs to the opponent. 
            // We pass -beta as their alpha, and -alpha as their beta.
            // Also, we invert the returned score to fit our perspective.
            score = -eval_ctx(next, depth - 1, -beta, -alpha, history, ply + 1, ctx, p);
        }

        delete next;

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

    for(auto& action : state->legal_actions){
        State* next = state->next_state(action);
        bool same = next->same_player_as_parent();

        int score;
        if(same){
            score = eval_ctx(next, depth - 1, alpha, beta, history, 1, ctx, p);
        } else {
            score = -eval_ctx(next, depth - 1, -beta, -alpha, history, 1, ctx, p);
        }

        delete next;

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

    result.score = best_score;
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
