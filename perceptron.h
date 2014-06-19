#include "stdafx.h"
#include "data.h"
#include "decoder.h"

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
