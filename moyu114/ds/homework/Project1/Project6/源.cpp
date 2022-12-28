#include<cstdio>
#include<iostream>
#include<queue>
#include<stack>
using namespace std;
void Reverse(queue<char>&qu) {
	stack<char>st;
	char temp;
	int n = qu.size();
	for (int i = 0;i < n;i++) {
		temp= qu.front();
		qu.pop();
		st.push(temp);
	}
	for (int i = 0;i < n;i++) {
		temp =st.top();
		st.pop();
		qu.push(temp);
	}
}
int main() {
	queue<char>qu;
	for (int i = 0;i < 6;i++) {
		qu.push('a' + i);
	}
	Reverse(qu);
	for (int i = 0;i < 6;i++) {
		cout << qu.front()<<" ";
		qu.pop();
	}
}