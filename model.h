#include "stdafx.h"
#include "datastruct.h"

class Model
{
	public:
		Model(const string &mode, const string &model_file, size_t line, size_t round);
		void load_validtagset();
		void parse_template();
		void load_model();
		void load_bin_model(const string &model_file);
		void save_model();
		void save_bin_model(size_t round);

		vector<int> get_validtagset(int cur_tok_id, int last_tag);
		double cal_local_score(const Cand &cand);
		void update_paras(const vector<int> &taglist_output, const vector<int> &taglist_gold, const size_t round, const size_t line);
		void update_paras_for_lastline(const size_t round, const size_t line);
		void set_cur_line_ptr(vector<vector<int> > *cur_line_ptr) {m_token_matrix_ptr=cur_line_ptr;};
	private:
		void update_paras_for_local_features(const vector<vector<int> > &local_features, const vector<vector<int> > &local_gold_features, const size_t round, const size_t line);
		vector<vector<int> > extract_features(const vector<int> &taglist,size_t feature_extract_pos);

	private:
		string MODE;
		size_t LINE;
		size_t ROUND;
		string m_model_file;
		vector<vector<int> > *m_token_matrix_ptr;
		unordered_map<vector<int>, WeightInfo, vechash> train_para_dict;
		unordered_map<vector<int>, double, vechash> test_para_dict;
		unordered_map<int, set<int> > tagset_for_token;
		unordered_map<int, set<int> > tagset_for_last_tag;
		vector<vector<pair<int,int> > > feature_templates;
		bool bigram_feature_flag;
		bool trigram_feature_flag;
};

