#include "decoder.h"

Decoder::Decoder(vector<vector<int> > *cur_line_ptr,Model *model)
{
	m_model = model;
	m_token_matrix_ptr = cur_line_ptr;
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
}

Decoder::~Decoder()
{
	m_model = NULL;
	m_token_matrix_ptr = NULL;
	candlist_old.resize(0);
	candlist_new.resize(0);
	m_gold_taglist.resize(0);
}

bool Decoder::decode_for_train(vector<int> &taglist_output, vector<int> &taglist_gold)
{
	for (cur_pos=2;cur_pos<m_token_matrix_ptr->size()-2;cur_pos++)
	{
		size_t len = m_token_matrix_ptr->at(cur_pos).size();
		m_gold_taglist.push_back(m_token_matrix_ptr->at(cur_pos).at(len-1));
		for (const auto &e_cand : candlist_old)
		{
			vector<Cand> candvec = expand(e_cand);
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
			taglist_output = candlist_old.at(0).taglist;
			taglist_gold = m_gold_taglist;
			return false;
		}
		//cout<<"decoding at pos "<<cur_pos-2<<endl;
	}
	if (candlist_old.at(0).taglist == m_gold_taglist)
		return true;
	taglist_output = candlist_old.at(0).taglist;
	taglist_gold = m_gold_taglist;
	return false;
}

vector<int> Decoder::decode()
{
	//cout<<"current sentence size: "<<m_token_matrix_ptr->size()-2<<endl;
	for (cur_pos=2;cur_pos<m_token_matrix_ptr->size()-2;cur_pos++)
	{
		for (const auto &e_cand : candlist_old)
		{
			vector<Cand> candvec = expand(e_cand);
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

vector<Cand> Decoder::expand(const Cand &cand)
{
	vector<Cand> candvec;
	int cur_tok_id = m_token_matrix_ptr->at(cur_pos).at(0);
	int last_tag = cand.taglist.at(cand.taglist.size()-1);
	vector<int> validtagset = m_model->get_validtagset(cur_tok_id,last_tag);

	for (const auto &e_tag : validtagset)
	{
		Cand cand_new;
		cand_new.taglist = cand.taglist;
		cand_new.taglist.push_back(e_tag);
		vector<vector<int> > local_features = extract_features(cand_new.taglist,cand_new.taglist.size()-1);
		double local_score = m_model->cal_local_score(local_features);
		cand_new.acc_score = cand.acc_score+local_score;
		candvec.push_back(cand_new);
	}
	return candvec;
}

bool Decoder::check_is_history_same(const Cand &cand0, const Cand &cand1)
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

void Decoder::add_to_new(const vector<Cand> &candvec)
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

vector<vector<int> > Decoder::extract_features(const vector<int> &taglist, size_t feature_extract_pos)
{
	vector<vector<int> > features;
	features.clear();
	vector<int> feature;
	int id = -1;
	for (const auto &ft : m_model->feature_templates)
	{
		id ++;
		feature.clear();
		feature.push_back(id);
		for(const auto &rcp : ft)
		{
			feature.push_back(m_token_matrix_ptr->at(feature_extract_pos+rcp.first).at(rcp.second));
		}
		feature.push_back(taglist.at(feature_extract_pos));
		features.push_back(feature);
	}
	if (m_model->bigram_feature_flag == true)
	{
		feature.clear();
		feature.push_back(++id);
		feature.push_back(taglist.at(feature_extract_pos-1));
		feature.push_back(taglist.at(feature_extract_pos));
		features.push_back(feature);
	}

	if (m_model->trigram_feature_flag == true)
	{
		feature.clear();
		feature.push_back(++id);
		feature.push_back(taglist.at(feature_extract_pos-2));
		feature.push_back(taglist.at(feature_extract_pos-1));
		feature.push_back(taglist.at(feature_extract_pos));
		features.push_back(feature);
	}
	return features;
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
