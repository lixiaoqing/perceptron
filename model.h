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
		double cal_local_score(const vector<vector<int> > &features);
		void update_paras_for_lastline(const size_t round, const size_t line);
		void update_paras(const vector<vector<int> > &local_features, const vector<vector<int> > &local_gold_features, const size_t round, const size_t line);

	public:
		vector<vector<pair<int,int> > > feature_templates;
		bool bigram_feature_flag;
		bool trigram_feature_flag;

	private:
		string MODE;
		size_t LINE;
		size_t ROUND;
		string m_model_file;
		unordered_map<vector<int>, WeightInfo, vechash> train_para_dict;
		unordered_map<vector<int>, double, vechash> test_para_dict;
		unordered_map<int, set<int> > tagset_for_token;
		unordered_map<int, set<int> > tagset_for_last_tag;
};

