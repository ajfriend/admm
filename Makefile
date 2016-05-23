.PHONY: test clean install

#all: test

# can run -vs, where s makes it not capture output
# the -l flag will print out a list of local variables with their corresponding values when a test fails
test:
	py.test admm -vs

clean:
	-pip uninstall admm
	-rm -rf build/ dist/ admm.egg-info/
	#-find . -name "*.cache" -exec rm -rf {} \;
	#-find . -name "__pycache__" -exec rm -rf {} \;
	-rm -rf __pycache__ admm/__pycache__ admm/tests/__pycache__ .cache
	-rm -rf .ipynb_checkpoints/

install:
	python setup.py install