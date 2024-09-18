// 'detectionRecipe.retrain'이라는 AngularJS 모듈을 생성
var app = angular.module('detectionRecipe.retrain', []);

// 'retrainRecipeController'라는 컨트롤러를 정의하고, $scope를 주입
app.controller('retrainRecipeController', function($scope) {
    
    // 최적화 방법 옵션을 정의
//    $scope.optimizerOptions = [
//        ["Adam", "adam"],
//        ["SGD", "sgd"]
//    ];

    // 특정 변수(varName)가 정의되지 않았을 경우 초기값(initValue)으로 설정하는 함수
    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue; // 변수의 초기값 설정
        }
    };

    // 데이터셋 정보를 가져오는 함수 정의
    var retrieveInfoRetrain = function() {
        // Python 메소드를 호출하여 데이터셋 정보를 가져옴
        $scope.callPythonDo({method: "get-dataset-info"}).then(function(data) {
            // GPU 사용 가능 여부 및 레이블 열 정보를 스코프에 설정
            $scope.canUseGPU = data["can_use_gpu"];
            $scope.labelColumns = data["columns"];
            
            // 기타 초기화 변수 설정
            initVariable("min_side", data["min_side"]);
            initVariable("max_side", data["max_side"]);
            
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        }, function(data) {
            $scope.finishedLoading = true; // 로딩 완료 상태로 설정
        });
    };

    // 여러 변수를 초기화하는 함수 정의
    var initVariables = function() {
        initVariable("val_split", 0.8); // 검증 데이터 비율 초기화
        initVariable("should_use_gpu", false); // GPU 사용 여부 초기화
        initVariable("gpu_allocation", 0.5); // GPU 할당 비율 초기화
        initVariable("list_gpu", "0"); // 사용할 GPU 목록 초기화
        initVariable("lr", 0.00001); // 학습률 초기화
        initVariable("nb_epochs", 10); // 에포크 수 초기화
        initVariable("tensorboard", false); // TensorBoard 사용 여부 초기화
        initVariable("optimizer", "adam"); // 최적화 방법 초기화
        initVariable("freeze", true); // 모델 동결 여부 초기화
        initVariable("epochs", 10); // 에포크 수 초기화
        initVariable("augment", true); // 데이터 증강 사용 여부 여부 초기화
        initVariable("reducelr", false); // 학습률 감소 사용 여부 초기화
        initVariable("reducelr_patience", 2); // 학습률 감소의 인내 기간 초기화
        initVariable("reducelr_factor", 0.1); // 학습률 감소 인자 초기화
        initVariable("single_column_data", false); // 단일 열 데이터 사용 여부 초기화
    };

    // 컨트롤러 초기화 함수 정의
    var init = function() {
        initVariables(); // 변수들 초기화
        retrieveInfoRetrain(); // 데이터셋 정보 가져오기
    };

    // 컨트롤러 초기화 함수 호출
    init();
});
