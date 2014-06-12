#include "stdafx.h"
#include "myutils.h"

struct Cand
{
	vector <int> taglist;
	double acc_score;
	bool in_track;
	bool operator> (const Cand &cand_rhs) const
	{
		return (acc_score > cand_rhs.acc_score);
	}
};

struct WeightInfo
{
	double weight;
	double acc_weight;
	size_t lastline;
	size_t lastround;
};

struct vechash 
{
	size_t operator()(const vector<int>& v) const
	{
		return hash_range(v.begin(),v.end());
	}
};

class Perceptron
{
	public:
		Perceptron();
		void train(string &train_file);
		void test(string &test_file);
		void set_mode(string mode) {MODE=mode;};
	private:
		void load_validtagset();
		void load_data(string &data_file);
		bool load_block(vector<vector<int> > &token_matrix, ifstream &fin);
		void decode_with_update();
		void save_model();
		void save_bin_model();

		void load_model();
		void load_bin_model();
		void decode();

		void expand(vector<Cand> &candvec, const Cand &cand);
		void extract_features(vector<vector<int> > &features, const vector<int> &taglist,size_t feature_extract_pos);
		void add_to_new(vector<Cand> &candlist);
		void update_paras();

	private:
		size_t ROUND;
		size_t LINE;
		size_t NGRAM;
		size_t BEAM_SIZE;
		string MODE;
		size_t m_line;
		size_t m_round;

		vector<vector<vector<int> > > m_token_matrix_list;
		vector<vector<int> > *m_token_matrix_ptr;
		vector<Cand> candlist_old;
		vector<Cand> candlist_new;
		vector<int> m_gold_taglist;
		size_t cur_pos;
		vector<vector<int> > local_features;
		vector<vector<int> > local_gold_features;

		unordered_map<vector<int>, WeightInfo, vechash> train_para_dict;
		unordered_map<vector<int>, double, vechash> test_para_dict;
		unordered_map<int, set<int> > tagset_for_token;
		unordered_map<int, set<int> > tagset_for_last_tag;
};
