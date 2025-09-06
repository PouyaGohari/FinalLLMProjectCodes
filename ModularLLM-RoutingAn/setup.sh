# git clone https://github.com/huggingface/peft.git
cd ./peft
echo "Change the directory to peft"
ls
# git fetch origin pull/2644/head:pr-2644
# git checkout pr-2644
pip uninstall peft
pip3 install -e . -q
cd ..
echo "Change the directory back to the content"
ls