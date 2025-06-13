git:
	git add .
	git commit -m "Update notebooks"
	git push

push: git
	kaggle kernels push -p notebooks

pull:
	kaggle kernels pull merongetaneh/CoTools -p notebooks

run:
	kaggle kernels output your-kaggle-username/your-kernel-name --wait

status:
	kaggle kernels status merongetaneh/CoTools

install:
	pip install -r package.txt

# kaggle kernels output your-kaggle-username/your-kernel-name --watch