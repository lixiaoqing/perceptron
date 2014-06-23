#include "model.h"

Model::Model(bool mode, const string &model_file, size_t line, size_t round)
{
	MODE = mode;
	LINE = line;
	ROUND = round;
	m_model_file = model_file;
	bigram_feature_flag = false;
	trigram_feature_flag = false;
	load_validtagset();
	parse_template();
	if (MODE == false)
	{
		load_bin_model(m_model_file);
	}
}

void Model::load_validtagset()
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

void Model::parse_template()
{
	ifstream fin;
	fin.open("template");
	if (!fin.is_open())
	{
		cerr<<"fail to open template file\n";
		return;
	}
	string sep("%");
	string line;
	while(getline(fin,line))
	{
		TrimLine(line);
		if (line.size() == 0)
			continue;
		if (line[0] == '#')
			continue;
		if (line == "B")
		{
			bigram_feature_flag = true;
			continue;
		}
		if (line == "T")
		{
			trigram_feature_flag = true;
			continue;
		}
		vector<pair<int,int> > feature_template;
		vector<string> toks;
		Split(toks,line,sep);
		for (size_t i=1;i<toks.size();i++)
		{
			string &tok = toks.at(i);
			tok = tok.substr(2,tok.size()-3);
			size_t p = tok.find(',');
			int row = stoi(tok.substr(0,p));
			int col = stoi(tok.substr(p+1));
			feature_template.push_back(make_pair(row,col));
		}
		feature_templates.push_back(feature_template);
	}
	fin.close();
}

void Model::save_model()
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
		fout<<fwp.second.acc_weight/LINE*ROUND<<endl;
	}
	cout<<"save model over\n";
}

void Model::save_bin_model(size_t round)
{
	ofstream fout(m_model_file+"."+to_string(round),ios::binary);
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
		double weight = fwp.second.acc_weight/(LINE*ROUND);
		fout.write((char*)&(weight),sizeof(double));
	}
	//cout<<"save binary model over\n";
}

void Model::load_model()
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

void Model::load_bin_model(const string &model_file)
{
	ifstream fin;
	fin.open(model_file.c_str(),ios::binary);
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

void Model::update_paras(const vector<vector<int> > &local_features, const vector<vector<int> > &local_gold_features, const size_t round, const size_t line)
{
	for (const auto &e_feature : local_features)
	{
		auto it = train_para_dict.find(e_feature);
		if (it == train_para_dict.end())
		{
			WeightInfo tmp = {-1,-1,line,round};
			train_para_dict.insert(make_pair(e_feature,tmp));
		}
		else
		{
			it->second.acc_weight += it->second.weight*((round - it->second.lastround)*LINE + line - it->second.lastline) - 1;
			it->second.weight += -1;
			it->second.lastline = line;
			it->second.lastround = round;
		}
	}

	for (const auto &e_feature : local_gold_features)
	{
		auto it = train_para_dict.find(e_feature);
		if (it == train_para_dict.end())
		{
			WeightInfo tmp = {1,1,line,round};
			train_para_dict.insert(make_pair(e_feature,tmp));
		}
		else
		{
			it->second.acc_weight += it->second.weight*((round-it->second.lastround)*LINE+line-it->second.lastline)+1;
			it->second.weight += 1;
			it->second.lastline = line;
			it->second.lastround = round;
		}
	}
}

void Model::update_paras_for_lastline(const size_t round, const size_t line)
{
	for (auto &fwp : train_para_dict)
	{
		fwp.second.acc_weight += fwp.second.weight*((round - fwp.second.lastround)*LINE + line - fwp.second.lastline);
		fwp.second.lastline = line;
		fwp.second.lastround = round;
	}
}

vector<int> Model::get_validtagset(int cur_tok_id, int last_tag)
{
	vector<int> validtagset;
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
	return validtagset;
}

double Model::cal_local_score(const vector<vector<int> > &features)
{
	double local_score = 0;
	for (const auto &e_feature : features)
	{
		if (MODE == true)
		{
			auto it=train_para_dict.find(e_feature);
			if (it != train_para_dict.end())
			{
				local_score += it->second.weight;
			}
		}
		else
		{
			auto it=test_para_dict.find(e_feature);
			if (it != test_para_dict.end())
			{
				local_score += it->second;
			}
		}
	}
	return local_score;
}


/*
vector<vector<int> > Model::extract_features(vector<vector<int> > &features, const vector<int> &taglist, size_t feature_extract_pos)
{
	vector<vector<int> > features;
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
	return features;
}
*/
