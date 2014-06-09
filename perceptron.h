#include "stdafx.h"
#include "myutils.h"

struct Cand
{
	vector <int> taglist;
	double acc_score;
	bool operator> (const Cand &cand_rhs) const
	{
		return (acc_score > cand_rhs.acc_score);
	}
};

struct WeightInfo
{
	double weight;
	double acc_weight;
	int lastline;
	int lastround;
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
		void decode_with_update();
		bool load_block(vector<vector<int> > &token_matrix, ifstream &fin);
		void save_model();

		void load_model();
		void decode();

		void expand(vector<Cand> &candvec, const Cand &cand);
		void extract_features(vector<vector<int> > &features, const vector<int> &taglist,size_t feature_extract_pos);
		void add_to_new(const vector<Cand> &candlist);
		void update_paras();

	private:
		size_t ROUND;
		size_t LINE;
		size_t NGRAM;
		size_t BEAM_SIZE;
		size_t m_line;
		size_t m_round;
		string MODE;
		vector<vector<vector<int> > > m_token_matrix_list;
		vector<vector<int> > *m_token_matrix_ptr;
		vector<int> m_gold_taglist;
		vector<vector<int> > local_features;
		vector<vector<int> > local_gold_features;
		size_t cur_pos;
		vector<Cand> candlist_old;
		vector<Cand> candlist_new;
		map<vector<int>, WeightInfo> train_para_dict;
		map<vector<int>, double> test_para_dict;
		map<int, set<int> > tagset_for_token;
		map<int, set<int> > tagset_for_last_tag;
};
