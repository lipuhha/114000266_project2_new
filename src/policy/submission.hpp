#pragma once
#include "search_types.hpp"
#include "game_history.hpp"

struct SubParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool report_partial = true;
    bool use_tt = true;
    bool use_quiescence = true;
    bool use_killer_moves = true;
    bool use_null_move = true;
    bool use_lmr = true;
    bool use_move_ordering = true;
    int quiescence_max_depth = 16;
    int null_move_r = 2;
    int lmr_full_depth = 3;
    int lmr_depth_limit = 3;
    int hash_size_mb = 64;

    static SubParams from_map(const ParamMap& m){
        SubParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        p.use_tt            = param_bool(m, "UseTT", true);
        p.use_quiescence    = param_bool(m, "UseQuiescence", true);
        p.use_killer_moves  = param_bool(m, "UseKillerMoves", true);
        p.use_null_move     = param_bool(m, "UseNullMove", true);
        p.use_lmr           = param_bool(m, "UseLMR", true);
        p.use_move_ordering = param_bool(m, "UseMoveOrdering", true);
        p.quiescence_max_depth = param_int(m, "QuiescenceMaxDepth", 16);
        p.null_move_r          = param_int(m, "NullMoveR", 2);
        p.lmr_full_depth       = param_int(m, "LMRFullDepth", 3);
        p.lmr_depth_limit      = param_int(m, "LMRDepthLimit", 3);
        p.hash_size_mb         = param_int(m, "Hash", 64);
        return p;
    }
};

class Submission{
public:
    // Notice the addition of alpha and beta parameters
    static int eval_ctx(
        State *state,
        int depth,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const SubParams& p,
        bool allow_null = true
    );
    
    static SearchResult search(
        State *state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
