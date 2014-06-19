#include "stdafx.h"
#include "myutils.h"

class Data
{
	public:
		void load_data(string &data_file);
		void shuffle(){random_shuffle(m_token_matrix_list.begin(),m_token_matrix_list.end());};
		size_t get_size() {return m_token_matrix_list.size();};
		vector<vector<int> >* get_token_matrix_ptr(size_t m_line) {return &m_token_matrix_list.at(m_line);};
		void set_mode(const string &mode){MODE=mode;};
		string get_tag(int id){return id2tag[id];};
	public:
		vector<vector<vector<int> > > m_token_matrix_list;
		vector<vector<string> > m_data_matrix;
	private:
		bool load_train_block(ifstream &fin);
		bool load_test_block(ifstream &fin);
		void save_dict();
		void load_dict();
	private:
		string MODE;
		size_t m_field_size;
		vector<int> ids;
		vector<unordered_map<string,int> > dict;
		unordered_map<int,set<int> > token2tagset;
		unordered_map<int,set<int> > lasttag2tagset;
		set<int> tagset;
		unordered_map<int,string> id2tag;
};
