#include "perceptron.h"

Perceptron::Perceptron()
{
	ROUND = 20;
	LINE = 1;
	m_line = 0;
	m_round = 0;
	m_token_matrix_list.clear();
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
		for (m_line=0;m_line<LINE;m_line++)
		{
			BeamSearchDecoder m_decoder(MODE,&(m_token_matrix_list.at(m_line)),this);
			size_t exit_pos;
			if(m_decoder.decode_for_train(exit_pos) == false)
			{
				for (size_t i=2;i<=exit_pos;i++)
				{
					vector<vector<int> > local_features;
					vector<vector<int> > local_gold_features;
					m_decoder.get_features_at_pos(local_features,local_gold_features,i);
					update_paras(local_features,local_gold_features);
				}
			}
			if (m_line == LINE - 1)
			{
				for (auto &fwp : train_para_dict)
				{
					fwp.second.acc_weight += fwp.second.weight*((m_round - fwp.second.lastround)*LINE + m_line - fwp.second.lastline);
					fwp.second.lastline = m_line;
					fwp.second.lastround = m_round;
				}
			}
			if (m_line%1000 == 0)
			{
				cout<<m_line<<" lines processed\n";
			}
		}
		cout<<"round "<<m_round<<endl;
		save_bin_model();
	}
	//save_model();
	//save_bin_model();
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
	for (const auto &fwp : train_para_dict)
	{
		if (fwp.second.acc_weight > -1e10 && fwp.second.acc_weight< 1e-10)
			continue;
		for (const auto &v : fwp.first)
		{
			fout<<v<<'\t';
		}
		fout<<fwp.second.acc_weight/(ROUND*LINE)<<endl;
	}
	cout<<"save model over\n";
}

void Perceptron::save_bin_model()
{
	ofstream fout("model.bin."+to_string(m_round),ios::binary);
	if (!fout.is_open())
	{
		cerr<<"fail to open binary model file to write!\n";
		return;
	}
	size_t feature_num = 0;
	for (const auto &fwp : train_para_dict)
	{
		if (fwp.second.acc_weight > -1e-10 && fwp.second.acc_weight < 1e-10)
			continue;
		feature_num++;
	}
	fout.write((char*)&feature_num,sizeof(size_t));
	for (const auto &fwp : train_para_dict)
	{
		if (fwp.second.acc_weight > -1e-10 && fwp.second.acc_weight < 1e-10)
			continue;
		size_t len = fwp.first.size();
		fout.write((char*)&len,sizeof(size_t));
		fout.write((char*)&(fwp.first.at(0)),sizeof(int)*fwp.first.size());
		double weight = fwp.second.acc_weight/(ROUND*LINE);
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
		vector<vector<int> > *cur_line_ptr = &(m_token_matrix_list.at(m_line));
		BeamSearchDecoder m_decoder(MODE,cur_line_ptr,this);
		vector<int> &output_taglist = m_decoder.decode();
		for (size_t i=2;i<output_taglist.size();i++)
		{
			fout<<cur_line_ptr->at(i).at(0)<<'\t'<<output_taglist.at(i)<<endl;
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
			validtagset.insert(stoi(toks.at(i)));
		}
		tagset_for_token[stoi(toks.at(0))] = validtagset;
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
			validtagset.insert(stoi(toks.at(i)));
		}
		tagset_for_last_tag[stoi(toks.at(0))] = validtagset;
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
		for (size_t i=0;i<fields.size();i++)
		{
			token_vec.push_back(stoi(fields.at(i)));
		}
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
			feature.push_back(stoi(toks.at(i)));
		}
		test_para_dict[feature] = stod(toks.at(toks.size()-1));
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

void Perceptron::update_paras(const vector<vector<int> > &local_features, const vector<vector<int> > &local_gold_features)
{
	for (const auto &e_feature : local_features)
	{
		auto it = train_para_dict.find(e_feature);
		if (it == train_para_dict.end())
		{
			WeightInfo tmp = {-1,-1,m_line,m_round};
			train_para_dict.insert(make_pair(e_feature,tmp));
		}
		else
		{
			it->second.acc_weight += it->second.weight*((m_round - it->second.lastround)*LINE + m_line - it->second.lastline) - 1;
			it->second.weight += -1;
			it->second.lastline = m_line;
			it->second.lastround = m_round;
		}
	}

	for (const auto &e_feature : local_gold_features)
	{
		auto it = train_para_dict.find(e_feature);
		if (it == train_para_dict.end())
		{
			WeightInfo tmp = {1,1,m_line,m_round};
			train_para_dict.insert(make_pair(e_feature,tmp));
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

void Perceptron::get_validtagset(vector<int> &validtagset, int cur_tok_id, int last_tag)
{
	validtagset.clear();
	set<int> validtagset1;
	auto it = tagset_for_token.find(cur_tok_id);
	if (it != tagset_for_token.end())
	{
		validtagset1 = it->second;
	}
	else
	{
		validtagset1 = tagset_for_token[-1];
	}

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
	set_intersection(validtagset1.begin(),validtagset1.end(),validtagset2.begin(),validtagset2.end(),back_inserter(validtagset));
}

double Perceptron::cal_local_score(const vector<vector<int> > &local_features)
{
	double local_score = 0;
	for (const auto &e_feature : local_features)
	{
		if (MODE == "train")
		{
			local_score += train_para_dict[e_feature].weight;
		}
		else
		{
			local_score += test_para_dict[e_feature];
		}
	}
	return local_score;
}

Perceptron::BeamSearchDecoder::BeamSearchDecoder(string &mode,vector<vector<int> > *cur_line_ptr,Perceptron *pcpt)
{
	m_pcpt = pcpt;
	MODE = mode;
	m_token_matrix_ptr = cur_line_ptr;
	candlist_old.clear();
	candlist_new.clear();
	Cand init_cand;
	init_cand.taglist.push_back(0);
	init_cand.taglist.push_back(0);
	init_cand.acc_score = 0;
	candlist_old.push_back(init_cand);
	if (mode == "train")
	{
		m_gold_taglist.clear();
		m_gold_taglist.push_back(0);
		m_gold_taglist.push_back(0);
	}
}

bool Perceptron::BeamSearchDecoder::decode_for_train(size_t &exit_pos)
{
	//cout<<"current sentence size: "<<m_token_matrix_ptr->size()-2<<endl;
	for (cur_pos=2;cur_pos<m_token_matrix_ptr->size()-2;cur_pos++)
	{
		size_t len = m_token_matrix_ptr->at(cur_pos).size();
		m_gold_taglist.push_back(m_token_matrix_ptr->at(cur_pos).at(len-1));
		for (const auto &e_cand : candlist_old)
		{
			vector<Cand> candvec;
			expand(candvec,e_cand);
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
		for (const auto &e_cand : candlist_old)
		{
			if (e_cand.taglist == m_gold_taglist)
			{
				lose_track = false;
				break;
			}
		}
		if (lose_track == true)
		{
			exit_pos = cur_pos;
			return false;
		}
		//cout<<"decoding at pos "<<cur_pos-2<<endl;
	}
	return true;
}

vector<int>& Perceptron::BeamSearchDecoder::decode()
{
	//cout<<"current sentence size: "<<m_token_matrix_ptr->size()-2<<endl;
	for (cur_pos=2;cur_pos<m_token_matrix_ptr->size()-2;cur_pos++)
	{
		size_t len = m_token_matrix_ptr->at(cur_pos).size();
		for (const auto &e_cand : candlist_old)
		{
			vector<Cand> candvec;
			expand(candvec,e_cand);
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
	return candlist_old.at(0).taglist;
}

void Perceptron::BeamSearchDecoder::expand(vector<Cand> &candvec, const Cand &cand)
{
	candvec.clear();
	int cur_tok_id = m_token_matrix_ptr->at(cur_pos).at(0);
	int last_tag = cand.taglist.at(cand.taglist.size()-1);
	vector<int> validtagset;
	m_pcpt->get_validtagset(validtagset,cur_tok_id,last_tag);

	for (const auto &e_tag : validtagset)
	{
		Cand cand_new;
		cand_new.taglist = cand.taglist;
		cand_new.taglist.push_back(e_tag);
		vector<vector<int> > m_local_features;
		extract_features(m_local_features,cand_new.taglist,cur_pos);
		double local_score = m_pcpt->cal_local_score(m_local_features);
		cand_new.acc_score = cand.acc_score+local_score;
		candvec.push_back(cand_new);
	}
}

bool Perceptron::BeamSearchDecoder::check_is_history_same(const Cand &cand0, const Cand &cand1)
{
	for (size_t k=0;k<NGRAM;k++)
	{
		if (cand0.taglist.at(cur_pos-k) != cand1.taglist.at(cur_pos-k))
		{
			return false;
		}
	}
	return true;
}

void Perceptron::BeamSearchDecoder::add_to_new(const vector<Cand> &candvec)
{
	for (const auto &e_cand : candvec)
	{
		bool is_history_same = false;
		/*
		*/
		for (auto &e_ori_cand : candlist_new)
		{
			is_history_same = check_is_history_same(e_cand,e_ori_cand);
			if (is_history_same == true)
			{
				if (e_cand.acc_score > e_ori_cand.acc_score)
				{
					e_ori_cand.taglist = e_cand.taglist;
					e_ori_cand.acc_score = e_cand.acc_score;
				}
				break;
			}
		}
		if (is_history_same == false)
		{
			candlist_new.push_back(e_cand);
		}
	}
}

/*
*/
void Perceptron::BeamSearchDecoder::extract_features(vector<vector<int> > &features, const vector<int> &taglist, size_t feature_extract_pos)
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

/*
void Perceptron::BeamSearchDecoder::extract_features(vector<vector<int> > &features, const vector<int> &taglist, size_t feature_extract_pos)
{
	features.clear();
	if (taglist.at(feature_extract_pos) == 0)
	{
		auto pw_begin_it = find(taglist.rend()-feature_extract_pos,taglist.rend(),0);
		size_t pw_begin = taglist.rend() - 1 - pw_begin_it;
		size_t pw_end = feature_extract_pos - 1;
		size_t ppw_begin = taglist.rend() -1 - find(pw_begin_it+1,taglist.rend(),0);
		size_t ppw_end = pw_begin-1;

		vector<int> feature;
		feature.push_back(1);
		for (size_t pos=pw_begin;pos<=pw_end;pos++)
		{
			feature.push_back(m_token_matrix_ptr->at(pos).at(0));
		}
		features.push_back(feature);

		size_t pw_len = pw_end - pw_begin + 1;
		if (pw_len == 1)
		{
			feature.at(0) = 2;
			features.push_back(feature);
		}

		feature.at(0) = 3;
		feature.push_back(m_token_matrix_ptr->at(feature_extract_pos).at(0));
		features.push_back(feature);

		feature.at(0) = 4;
		feature.at(feature.size()-1) = m_token_matrix_ptr->at(ppw_end).at(0);
		features.push_back(feature);

		size_t ppw_len = ppw_end - ppw_begin + 1;
		feature.at(0) = 5;
		feature.at(feature.size()-1) = ppw_len;
		features.push_back(feature);

		feature.at(0) = 6;
		feature.at(feature.size()-1) = 0;
		for (size_t pos=ppw_begin;pos<=ppw_end;pos++)
		{
			feature.push_back(m_token_matrix_ptr->at(pos).at(0));
		}
		features.push_back(feature);

		feature.clear();
		feature.push_back(7);
		feature.push_back(m_token_matrix_ptr->at(pw_begin).at(0));
		feature.push_back(pw_len);
		features.push_back(feature);

		feature.at(0) = 8;
		feature.at(1) = m_token_matrix_ptr->at(pw_end).at(0);
		features.push_back(feature);

		feature.at(0) = 9;
		feature.at(2) = m_token_matrix_ptr->at(feature_extract_pos).at(0);
		features.push_back(feature);

		feature.at(0) = 10;
		feature.at(1) = m_token_matrix_ptr->at(pw_begin).at(0);
		feature.at(2) = m_token_matrix_ptr->at(pw_end).at(0);
		features.push_back(feature);

		feature.at(0) = 11;
		feature.at(2) = m_token_matrix_ptr->at(feature_extract_pos).at(0);
		features.push_back(feature);

		feature.at(0) = 12;
		feature.at(1) = m_token_matrix_ptr->at(ppw_end).at(0);
		feature.at(2) = m_token_matrix_ptr->at(pw_end).at(0);
		features.push_back(feature);

		feature.clear();
		feature.push_back(13);
		for (size_t pos=ppw_begin;pos<=ppw_end;pos++)
		{
			feature.push_back(m_token_matrix_ptr->at(pos).at(0));
		}
		feature.push_back(pw_len);
		features.push_back(feature);
	}
	else
	{
		vector<int> feature;
		feature.push_back(14);
		feature.push_back(m_token_matrix_ptr->at(feature_extract_pos-1).at(0));
		feature.push_back(m_token_matrix_ptr->at(feature_extract_pos).at(0));
		features.push_back(feature);
	}
}
*/

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
