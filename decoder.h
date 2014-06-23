#include "stdafx.h"
#include "model.h"

class Decoder
{
	public:
		Decoder(vector<vector<int> > *cur_line_ptr,Model *model);
		~Decoder();
		bool decode_for_train(vector<int> &taglist_output, vector<int> &taglist_gold);
		vector<int> decode();
		vector<vector<int> > extract_features(const vector<int> &taglist,size_t feature_extract_pos);

	private:
		vector<Cand> expand(const Cand &cand);
		void add_to_new(const vector<Cand> &candlist);
		bool check_is_history_same(const Cand &cand0, const Cand &cand1);

	private:
		Model *m_model;
		vector<vector<int> > *m_token_matrix_ptr;
		const static size_t BEAM_SIZE = 16;
		const static size_t NGRAM = 3;
		vector<Cand> candlist_old;
		vector<Cand> candlist_new;
		vector<int> m_gold_taglist;
		size_t cur_pos;
};
