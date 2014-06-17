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

class Model
{
	public:
		Model();
		vector<int> get_validtagset(int cur_tok_id, int last_tag);
		double cal_local_score(const vector<vector<int> > &local_features);
		void load_validtagset();
		void save_model(size_t total_line);
		void save_bin_model(size_t round, size_t total_line);
		void load_model();
		void load_bin_model();

		void update_paras(const vector<vector<int> > &local_features, const vector<vector<int> > &local_gold_features, const size_t round, const size_t line);
		void update_paras_for_lastline(const size_t round, const size_t line);
		void set_line(size_t line){LINE = line;};
		void set_mode(const string &mode){MODE=mode;};

	private:
		size_t LINE;
		string MODE;
		unordered_map<vector<int>, WeightInfo, vechash> train_para_dict;
		unordered_map<vector<int>, double, vechash> test_para_dict;
		unordered_map<int, set<int> > tagset_for_token;
		unordered_map<int, set<int> > tagset_for_last_tag;
};

class Data
{
	public:
		void load_data(string &data_file);
		bool load_block(vector<vector<int> > &token_matrix, ifstream &fin);
		void shuffle(){random_shuffle(m_token_matrix_list.begin(),m_token_matrix_list.end());};
		size_t get_size() {return m_token_matrix_list.size();};
		vector<vector<int> >* get_token_matrix_ptr(size_t m_line) {return &m_token_matrix_list.at(m_line);};
	public:
		vector<vector<vector<int> > > m_token_matrix_list;
};

class Decoder
{
	public:
		Decoder(vector<vector<int> > *cur_line_ptr,Model *model);
		bool decode_for_train(size_t &exit_pos);
		vector<int> decode();

		void get_features_at_pos(vector<vector<int> > &local_features,vector<vector<int> > &local_gold_features,size_t pos) {local_features = extract_features(candlist_old.at(0).taglist,pos);local_gold_features = extract_features(m_gold_taglist,pos);};
	private:
		vector<vector<int> > extract_features(const vector<int> &taglist,size_t feature_extract_pos);
		vector<Cand> expand(const Cand &cand);
		void add_to_new(const vector<Cand> &candlist);
		bool check_is_history_same(const Cand &cand0, const Cand &cand1);

	private:
		Model *m_model;
		const static size_t BEAM_SIZE = 16;
		const static size_t NGRAM = 2;
		vector<vector<int> > *m_token_matrix_ptr;
		vector<Cand> candlist_old;
		vector<Cand> candlist_new;
		vector<int> m_gold_taglist;
		size_t cur_pos;
};

class Perceptron
{
	public:
		Perceptron(Data *data,Model *model);
		void train(string &train_file);
		void test(string &test_file);

	private:
		size_t ROUND;
		size_t LINE;
		size_t m_line;
		size_t m_round;
		Model *m_model;
		Data *m_data;
};
