#include "data.h"

void Data::save_dict()
{
	ofstream fout;
	fout.open("dict");
	if (!fout.is_open())
	{
		cerr<<"fail to open dict file to write\n";
		return;
	}
	for (const auto &d : dict)
	{
		fout<<"###dict###"<<endl;
		for (const auto &kvp : d)
		{
			fout<<kvp.first<<'\t'<<kvp.second<<endl;
		}
	}
	fout.close();

	fout.open("tagset_for_token");
	if (!fout.is_open())
	{
		cerr<<"fail to open tagset_for_token file to write\n";
		return;
	}
	fout<<"-1";
	for (const auto &e : tagset)
	{
		fout<<' '<<e;
	}
	/*
	fout<<endl;
	for (const auto &kvp : token2tagset)
	{
		fout<<kvp.first;
		for (const auto &e : kvp.second)
		{
			fout<<' '<<e;
		}
		fout<<endl;
	}
	*/
	fout.close();

	fout.open("tagset_for_last_tag");
	if (!fout.is_open())
	{
		cerr<<"fail to open tagset_for_last_tag file to write\n";
		return;
	}
	fout<<"-1";
	for (const auto &e : tagset)
	{
		fout<<' '<<e;
	}
	fout<<endl;
	for (const auto &kvp : lasttag2tagset)
	{
		fout<<kvp.first;
		for (const auto &e : kvp.second)
		{
			fout<<' '<<e;
		}
		fout<<endl;
	}
	fout.close();
}

void Data::load_dict()
{
	dict.resize(m_field_size+1);
	ifstream fin;
	fin.open("dict");
	if (!fin.is_open())
	{
		cerr<<"fail to open dict file to load\n";
		return;
	}

	int dict_id = -1;
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		if (line == "###dict###")
		{
			dict_id += 1;
		}
		else
		{
			vector<string> kv;
			Split(kv,line);
			dict.at(dict_id).insert(make_pair(kv.at(0),stoi(kv.at(1))));
		}
	}
	fin.close();
	for(const auto &kv : dict.at(dict_id))
	{
		id2tag.insert(make_pair(kv.second,kv.first));
	}
}

void Data::load_data(string &data_file)
{
	ifstream fin;
	fin.open(data_file.c_str());
	if (!fin.is_open())
	{
		cerr<<"fail to open data file!\n";
		return;
	}
	string line;
	getline(fin,line);
	vector<string> fields;
	Split(fields,line);
	m_field_size = fields.size();
	fin.seekg(0,fin.beg);

	if (MODE == "train")
	{
		ids.resize(m_field_size,1);
		dict.resize(m_field_size);
		while(load_train_block(fin));
		save_dict();
	}
	else
	{
		load_dict();
		while(load_test_block(fin));
	}
	cout<<"load data over\n";
}

bool Data::load_test_block(ifstream &fin)
{
	vector<vector<int> > token_matrix;
	vector<string> line_list(2,"<s>");
	string line;
	vector<int> default_token_vec(m_field_size,0);
	token_matrix.push_back(default_token_vec);
	token_matrix.push_back(default_token_vec);
	while(getline(fin,line))
	{
		TrimLine(line);
		if (line.size() == 0)
		{
			token_matrix.push_back(default_token_vec);
			token_matrix.push_back(default_token_vec);
			m_token_matrix_list.push_back(token_matrix);
			m_data_matrix.push_back(line_list);
			return true;
		}
		vector<string> fields;
		Split(fields,line);
		vector<int> token_vec;
		for (size_t i=0;i<m_field_size;i++)
		{
			string &field = fields.at(i);
			auto it=dict.at(i).find(field);
			if (it != dict.at(i).end())
			{
				token_vec.push_back(it->second);
			}
			else
			{
				token_vec.push_back(-1);
			}
		}
		token_matrix.push_back(token_vec);
		line_list.push_back(line);
	}
	return false;
}

bool Data::load_train_block(ifstream &fin)
{
	vector<vector<int> > token_matrix;
	string line;
	vector<int> default_token_vec(m_field_size,0);
	token_matrix.push_back(default_token_vec);
	token_matrix.push_back(default_token_vec);
	size_t lasttag_id = 0;
	while(getline(fin,line))
	{
		TrimLine(line);
		if (line.size() == 0)
		{
			token_matrix.push_back(default_token_vec);
			token_matrix.push_back(default_token_vec);
			m_token_matrix_list.push_back(token_matrix);
			return true;
		}
		vector<string> fields;
		Split(fields,line);
		vector<int> token_vec;
		for (size_t i=0;i<m_field_size;i++)
		{
			string &field = fields.at(i);
			auto it=dict.at(i).find(field);
			if (it != dict.at(i).end())
			{
				token_vec.push_back(it->second);
			}
			else
			{
				dict.at(i).insert(make_pair(field,ids.at(i)));
				token_vec.push_back(ids.at(i));
				ids.at(i) += 1;
			}
		}
		token2tagset[token_vec.at(0)].insert(token_vec.at(m_field_size-1));
		lasttag2tagset[lasttag_id].insert(token_vec.at(m_field_size-1));
		tagset.insert(token_vec.at(m_field_size-1));
		lasttag_id = token_vec.at(m_field_size-1);
		/*
		for (auto &e_field : fields)
		{
			token_vec.push_back(stoi(e_field));
		}
		*/
		token_matrix.push_back(token_vec);
	}
	return false;
}
