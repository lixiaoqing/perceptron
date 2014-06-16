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
	size_t lastline;
	size_t lastround;
};

struct vechash 
{
	size_t operator()(const vector<int>& v) const
	{
		//return hash_range(v.begin(),v.end());
		return fnv1_hash(v);
		//return djb2_hash(v);
		//return bkdr_hash(v);
	}
};

class Perceptron
{
	public:
		Perceptron();
		void train(string &train_file);
		void test(string &test_file);
		void set_mode(string mode) {MODE=mode;};

		friend class BeamSearchDecoder;

	private:
		void load_validtagset();
		void load_data(string &data_file);
		bool load_block(vector<vector<int> > &token_matrix, ifstream &fin);
		void save_model();
		void save_bin_model();

		void load_model();
		void load_bin_model();

		void update_paras(const vector<vector<int> > &local_features, const vector<vector<int> > &local_gold_features);

		void get_validtagset(vector<int> &validtagset, int cur_tok_id, int last_tag);
		double cal_local_score(const vector<vector<int> > &local_features);

	private:
		string MODE;
		size_t ROUND;
		size_t LINE;
		size_t m_line;
		size_t m_round;

		vector<vector<vector<int> > > m_token_matrix_list;

		unordered_map<vector<int>, WeightInfo, vechash> train_para_dict;
		unordered_map<vector<int>, double, vechash> test_para_dict;
		unordered_map<int, set<int> > tagset_for_token;
		unordered_map<int, set<int> > tagset_for_last_tag;

	private:
		class BeamSearchDecoder
		{
			public:
				BeamSearchDecoder(string &mode,vector<vector<int> > *cur_line_ptr,Perceptron *pcpt);
				bool decode_for_train(size_t &exit_pos);
				vector<int> decode();

				void get_features_at_pos(vector<vector<int> > &local_features,vector<vector<int> > &local_gold_features,size_t pos) {local_features = extract_features(candlist_old.at(0).taglist,pos);local_gold_features = extract_features(m_gold_taglist,pos);};
			private:
				vector<vector<int> > extract_features(const vector<int> &taglist,size_t feature_extract_pos);
				vector<Cand> expand(const Cand &cand);
				void add_to_new(const vector<Cand> &candlist);
				bool check_is_history_same(const Cand &cand0, const Cand &cand1);

			private:
				Perceptron *m_pcpt;
				const static size_t BEAM_SIZE = 16;
				const static size_t NGRAM = 2;
				string MODE;
				vector<vector<int> > *m_token_matrix_ptr;
				vector<Cand> candlist_old;
				vector<Cand> candlist_new;
				vector<int> m_gold_taglist;
				size_t cur_pos;
		};
};

