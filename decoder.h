#include "stdafx.h"
#include "model.h"

class Decoder
{
	public:
		Decoder(vector<vector<int> > *cur_line_ptr,Model *model,bool mode);
		~Decoder();
		vector<int> decode();
		vector<vector<int> > extract_features(const vector<int> &taglist,size_t feature_extract_pos);

	private:
		vector<Cand> expand(const Cand &cand);
		void add_to_new(const vector<Cand> &candlist);
		bool check_is_history_same(const Cand &cand0, const Cand &cand1);

	public:
		vector<int> m_gold_taglist;
	private:
		Model *m_model;
		vector<vector<int> > *m_token_matrix_ptr;
		bool MODE;
		const static size_t BEAM_SIZE = 16;
		const static size_t NGRAM = 1;
		vector<Cand> candlist_old;
		vector<Cand> candlist_new;
		size_t cur_pos;
};
