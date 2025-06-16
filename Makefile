# Define your actual Kaggle username and kernel name here
USERNAME=merongetaneh
KERNEL_NAME=CoTools

git:
	git add .
	git commit -m "Update notebooks"
	git push

push: git
	kaggle kernels push -p notebooks

pull:
	kaggle kernels pull $(USERNAME)/$(KERNEL_NAME) -p notebooks -m

run:
	kaggle kernels output $(USERNAME)/$(KERNEL_NAME) --wait

watch:
	kaggle kernels output $(USERNAME)/$(KERNEL_NAME) --watch

status:
	kaggle kernels status $(USERNAME)/$(KERNEL_NAME)

install:
	pip install -r package.txt
