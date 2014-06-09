#include "perceptron.h"

Perceptron::Perceptron()
{
	ROUND = 10;
	NGRAM = 3;
	LINE = 1;
	BEAM_SIZE = 16;
	m_line = 0;
	m_round = 0;
	m_token_matrix_list.clear();
	m_token_matrix_ptr = NULL;
	m_gold_taglist.clear();
	candlist_old.clear();
	candlist_new.clear();
	train_para_dict.clear();
	test_para_dict.clear();
	tagset_for_token.clear();
	tagset_for_last_tag.clear();
	load_validtagset();
}

void Perceptron::train(string &train_file)
{
	load_data(train_file);
	LINE = m_token_matrix_list.size();
	for (m_round=0;m_round<ROUND;m_round++)
	{
		random_shuffle(m_token_matrix_list.begin(),m_token_matrix_list.end());
		for(m_line=0;m_line<LINE;m_line++)
		{
			m_token_matrix_ptr = &(m_token_matrix_list.at(m_line));
			decode_with_update();
			if (m_line == LINE - 1)
			{
				for (map<vector<int>,WeightInfo>::iterator it=train_para_dict.begin();it!=train_para_dict.end();it++)
				{
					it->second.acc_weight += it->second.weight*((m_round - it->second.lastround)*LINE + m_line - it->second.lastline);
					it->second.lastline = m_line;
					it->second.lastround = m_round;
				}
			}
			if (m_line%1000 == 0)
			{
				cout<<m_line<<" lines processed\n";
			}
		}
		cout<<"round "<<m_round<<endl;
	}
	//save_model();
	save_bin_model();
}

void Perceptron::save_model()
{
	ofstream fout;
	fout.open("model");
	if (!fout.is_open())
	{
		cerr<<"fail to open model file to write\n";
		return;
	}
	for(map<vector<int>,WeightInfo>::iterator it=train_para_dict.begin();it!=train_para_dict.end();it++)
	{
		if (it->second.acc_weight > -1e-10 && it->second.acc_weight < 1e-10)
			continue;
		for (size_t i=0;i<it->first.size();i++)
		{
			fout<<it->first.at(i)<<'\t';
		}
		fout<<it->second.acc_weight/(ROUND*LINE)<<endl;
	}
	cout<<"save model over\n";
}

void Perceptron::save_bin_model()
{
	ofstream fout("model.bin",ios::binary);
	if (!fout.is_open())
	{
		cerr<<"fail to open binary model file to write!\n";
		return;
	}
	size_t feature_num = 0;
	for(map<vector<int>,WeightInfo>::iterator it=train_para_dict.begin();it!=train_para_dict.end();it++)
	{
		if (it->second.acc_weight > -1e-10 && it->second.acc_weight < 1e-10)
			continue;
		feature_num++;
	}
	fout.write((char*)&feature_num,sizeof(size_t));
	for(map<vector<int>,WeightInfo>::iterator it=train_para_dict.begin();it!=train_para_dict.end();it++)
	{
		if (it->second.acc_weight > -1e-10 && it->second.acc_weight < 1e-10)
			continue;
		size_t len = it->first.size();
		fout.write((char*)&len,sizeof(size_t));
		fout.write((char*)&(it->first.at(0)),sizeof(int)*it->first.size());
		double weight = it->second.acc_weight/(ROUND*LINE);
		fout.write((char*)&(weight),sizeof(double));
	}
	cout<<"save binary model over\n";
}

void Perceptron::test(string &test_file)
{
	load_bin_model();
	load_data(test_file);
	LINE = m_token_matrix_list.size();
	ofstream fout;
	fout.open("output");
	if (!fout.is_open())
	{
		cerr<<"fail to open output file\n";
		return;
	}

	for(size_t m_line=0;m_line<LINE;m_line++)
	{
		m_token_matrix_ptr = &(m_token_matrix_list.at(m_line));
		decode();
		vector<int> &output_taglist = candlist_old.at(0).taglist;
		for (size_t i=2;i<output_taglist.size();i++)
		{
			fout<<m_token_matrix_ptr->at(i).at(0)<<'\t'<<output_taglist.at(i)<<endl;
		}
		fout<<endl;
	}
}

void Perceptron::load_validtagset()
{
	ifstream fin;
	fin.open("tagset_for_token");
	if (!fin.is_open())
	{
		cerr<<"fail to open tagset_for_token file\n";
		return;
	}
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		vector<string> toks;
		Split(toks,line);
		set<int> validtagset;
		for (size_t i=1;i<toks.size();i++)
		{
			validtagset.insert(s2i(toks.at(i)));
		}
		tagset_for_token[s2i(toks.at(0))] = validtagset;
	}
	fin.close();

	fin.open("tagset_for_last_tag");
	if (!fin.is_open())
	{
		cerr<<"fail to open tagset_for_last_tag file\n";
		return;
	}
	while(getline(fin,line))
	{
		TrimLine(line);
		vector<string> toks;
		Split(toks,line);
		set<int> validtagset;
		for (size_t i=1;i<toks.size();i++)
		{
			validtagset.insert(s2i(toks.at(i)));
		}
		tagset_for_last_tag[s2i(toks.at(0))] = validtagset;
	}
	fin.close();
}

void Perceptron::load_data(string &data_file)
{
	ifstream fin;
	fin.open(data_file.c_str());
	if (!fin.is_open())
	{
		cerr<<"fail to open data file!\n";
		return;
	}
	vector<vector<int> > token_matrix;
	while(load_block(token_matrix,fin))
	{
		m_token_matrix_list.push_back(token_matrix);
	}
	cout<<"load data over\n";
}

bool Perceptron::load_block(vector<vector<int> > &token_matrix, ifstream &fin)
{
	token_matrix.clear();
	string line;
	vector<int> default_token_vec;
	token_matrix.push_back(default_token_vec);
	token_matrix.push_back(default_token_vec);
	size_t field_size;
	while(getline(fin,line))
	{
		TrimLine(line);
		if (line.size() == 0)
		{
			token_matrix.at(0).resize(field_size,0);
			token_matrix.at(1).resize(field_size,0);
			token_matrix.push_back(token_matrix.at(0));
			token_matrix.push_back(token_matrix.at(0));
			return true;
		}
		vector<string> fields;
		Split(fields,line);
		field_size = fields.size();
		vector<int> token_vec;
		for (size_t i=0;i<field_size;i++)
		{
			token_vec.push_back(s2i(fields.at(i)));
		}
		token_matrix.push_back(token_vec);
	}
	return false;
}

void Perceptron::decode_with_update()
{
	//init 
	candlist_old.clear();
	candlist_new.clear();
	Cand init_cand;
	init_cand.taglist.push_back(0);
	init_cand.taglist.push_back(0);
	init_cand.acc_score = 0;
	candlist_old.push_back(init_cand);
	m_gold_taglist.clear();
	m_gold_taglist.push_back(0);
	m_gold_taglist.push_back(0);

	//cout<<"current sentence size: "<<m_token_matrix_ptr->size()-2<<endl;
	for (cur_pos=2;cur_pos<m_token_matrix_ptr->size()-2;cur_pos++)
	{
		size_t len = m_token_matrix_ptr->at(cur_pos).size();
		m_gold_taglist.push_back(m_token_matrix_ptr->at(cur_pos).at(len-1));
		for (size_t i=0;i<candlist_old.size();i++)
		{
			vector<Cand> candvec;
			expand(candvec,candlist_old.at(i));
			add_to_new(candvec);
		}

		sort(candlist_new.begin(),candlist_new.end(),greater<Cand>());

		candlist_old.swap(candlist_new);
		if (candlist_old.size() > BEAM_SIZE)
		{
			candlist_old.resize(BEAM_SIZE);
		}
		candlist_new.resize(0);

		bool lose_track = true;
		for (size_t i=0;i<candlist_old.size();i++)
		{
			if (candlist_old.at(i).taglist == m_gold_taglist)
			{
				lose_track = false;
				break;
			}
		}
		if (lose_track == true)
		{
			for (size_t i=2;i<=cur_pos;i++)
			{
				extract_features(local_features,candlist_old.at(0).taglist,i);
				extract_features(local_gold_features,m_gold_taglist,i);
				update_paras();
			}
			break;
		}
		//cout<<"decoding at pos "<<cur_pos-2<<endl;
	}
}

void Perceptron::load_model()
{
	ifstream fin;
	fin.open("model");
	if (!fin.is_open())
	{
		cerr<<"fail to open model file\n";
		return;
	}
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		vector<string> toks;
		Split(toks,line);
		vector<int> feature;
		for (size_t i=0;i<toks.size()-1;i++)
		{
			feature.push_back(s2i(toks.at(i)));
		}
		test_para_dict[feature] = s2d(toks.at(toks.size()-1));
	}
	cout<<"load model over\n";
}

void Perceptron::load_bin_model()
{
	ifstream fin;
	fin.open("model.bin",ios::binary);
	if (!fin.is_open())
	{
		cerr<<"fail to open binary model file\n";
		return;
	}
	size_t feature_num;
	fin.read((char*)&feature_num, sizeof(size_t));
	size_t len;
	for (size_t i = 0; i < feature_num; i++) 
	{
		fin.read((char*)&len, sizeof(size_t));
		vector<int> feature;
		feature.resize(len);
		fin.read((char*)&(feature.at(0)),sizeof(int)*len);
		double weight;
		fin.read((char*)&weight, sizeof(double));
		test_para_dict[feature] = weight;
	}
	cout<<"load binary model over\n";
}

void Perceptron::decode()
{
	candlist_old.clear();
	candlist_new.clear();
	Cand init_cand;
	init_cand.taglist.push_back(0);
	init_cand.taglist.push_back(0);
	init_cand.acc_score = 0;
	candlist_old.push_back(init_cand);

	//cout<<"current sentence size: "<<m_token_matrix_ptr->size()-2<<endl;
	for (cur_pos=2;cur_pos<m_token_matrix_ptr->size()-2;cur_pos++)
	{
		size_t len = m_token_matrix_ptr->at(cur_pos).size();
		for (size_t i=0;i<candlist_old.size();i++)
		{
			vector<Cand> candvec;
			expand(candvec,candlist_old.at(i));
			add_to_new(candvec);
		}

		sort(candlist_new.begin(),candlist_new.end(),greater<Cand>());

		candlist_old.swap(candlist_new);
		if (candlist_old.size() > BEAM_SIZE)
		{
			candlist_old.resize(BEAM_SIZE);
		}
		candlist_new.resize(0);
		//cout<<"decoding at pos "<<cur_pos-2<<endl;
	}
}

void Perceptron::expand(vector<Cand> &candvec, const Cand &cand)
{
	candvec.clear();
	int cur_tok_id = m_token_matrix_ptr->at(cur_pos).at(0);
	set<int> validtagset1;
	map<int,set<int> >::iterator it = tagset_for_token.find(cur_tok_id);
	if (it != tagset_for_token.end())
	{
		validtagset1 = it->second;
	}
	else
	{
		validtagset1 = tagset_for_token[-1];
	}

	int last_tag = cand.taglist.at(cand.taglist.size()-1);
	set<int> validtagset2;
	it = tagset_for_last_tag.find(last_tag);
	if (it != tagset_for_last_tag.end())
	{
		validtagset2 = it->second;
	}
	else
	{
		validtagset2 = tagset_for_last_tag[-1];
	}

	vector<int> validtagset;
	set_intersection(validtagset1.begin(),validtagset1.end(),validtagset2.begin(),validtagset2.end(),back_inserter(validtagset));
	for (size_t i=0;i<validtagset.size();i++)
	{
		Cand cand_new;
		cand_new.taglist = cand.taglist;
		cand_new.taglist.push_back(validtagset.at(i));
		double local_score = 0;
		extract_features(local_features,cand_new.taglist,cur_pos);
		for (size_t j=0;j<local_features.size();j++)
		{
			if (MODE == "train")
			{
				local_score += train_para_dict[local_features.at(j)].acc_weight;
			}
			else
			{
				local_score += test_para_dict[local_features.at(j)];
			}
		}
		cand_new.acc_score = cand.acc_score+local_score;
		candvec.push_back(cand_new);
	}
}

void Perceptron::extract_features(vector<vector<int> > &features, const vector<int> &taglist, size_t feature_extract_pos)
{
	features.clear();
	int arr[] = {-2,-1,0,1,2};
	vector<int> feature;
	for (size_t i=0;i<5;i++)
	{
		feature.clear();
		feature.push_back(i);
		feature.push_back(m_token_matrix_ptr->at(feature_extract_pos+arr[i]).at(0));
		feature.push_back(taglist.at(feature_extract_pos));
		features.push_back(feature);
	}
	for (size_t i=0;i<4;i++)
	{
		feature.clear();
		feature.push_back(i+5);
		feature.push_back(m_token_matrix_ptr->at(feature_extract_pos+arr[i]).at(0));
		feature.push_back(m_token_matrix_ptr->at(feature_extract_pos+arr[i+1]).at(0));
		feature.push_back(taglist.at(feature_extract_pos));
		features.push_back(feature);
	}
	feature.clear();
	feature.push_back(9);
	feature.push_back(m_token_matrix_ptr->at(feature_extract_pos-1).at(0));
	feature.push_back(m_token_matrix_ptr->at(feature_extract_pos+1).at(0));
	feature.push_back(taglist.at(feature_extract_pos));
	features.push_back(feature);
}

void Perceptron::add_to_new(const vector<Cand> &candvec)
{
	for (size_t i=0;i<candvec.size();i++)
	{
		bool is_history_same = false;
		for (size_t j=0;j<candlist_new.size();j++)
		{
			is_history_same = true;
			for (size_t k=0;k<NGRAM;k++)
			{
				if (candvec.at(i).taglist.at(cur_pos-k) != candlist_new.at(j).taglist.at(cur_pos-k))
				{
					is_history_same = false;
					break;
				}
			}
			if (is_history_same == true)
			{
				if (candvec.at(i).acc_score > candlist_new.at(j).acc_score)
				{
					candlist_new.at(j).taglist = candvec.at(i).taglist;
					candlist_new.at(j).acc_score = candvec.at(i).acc_score;
				}
				break;
			}
		}
		if (is_history_same == false)
		{
			candlist_new.push_back(candvec.at(i));
		}
	}
}

void Perceptron::update_paras()
{
	for (size_t i=0;i<local_features.size();i++)
	{
		map<vector<int>,WeightInfo>::iterator it = train_para_dict.find(local_features.at(i));
		if (it == train_para_dict.end())
		{
			WeightInfo tmp = {-1,-1,m_line,m_round};
			train_para_dict.insert(make_pair(local_features.at(i),tmp));
		}
		else
		{
			it->second.acc_weight += it->second.weight*((m_round - it->second.lastround)*LINE + m_line - it->second.lastline) - 1;
			it->second.weight += -1;
			it->second.lastline = m_line;
			it->second.lastround = m_round;
		}
	}
	for (size_t i=0;i<local_gold_features.size();i++)
	{
		map<vector<int>,WeightInfo>::iterator it = train_para_dict.find(local_gold_features.at(i));
		if (it == train_para_dict.end())
		{
			WeightInfo tmp = {1,1,m_line,m_round};
			train_para_dict.insert(make_pair(local_gold_features.at(i),tmp));
		}
		else
		{
			it->second.acc_weight += it->second.weight*((m_round-it->second.lastround)*LINE+m_line-it->second.lastline)+1;
			it->second.weight += 1;
			it->second.lastline = m_line;
			it->second.lastround = m_round;
		}
	}
}

int main(int argc, char *argv[])
{
	clock_t a,b;
	a = clock();

	Perceptron my_pcpt;
	if (argv[1][1] == 't')
	{
		my_pcpt.set_mode("train");
		string train_file(argv[2]);
		my_pcpt.train(train_file);
	}
	else if (argv[1][1] == 'd')
	{
		my_pcpt.set_mode("test");
		string test_file(argv[2]);
		my_pcpt.test(test_file);
	}

	b = clock();
	cout<<"time cost: "<<double(b-a)/CLOCKS_PER_SEC;
	return 0;
}
