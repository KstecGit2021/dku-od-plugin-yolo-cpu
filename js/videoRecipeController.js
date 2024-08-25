var app = angular.module('detectionRecipe.video', []);

app.controller('videoRecipeController', function($scope) {


    var retrieveCanUseGPU = function() {

        $scope.callPythonDo({method: "get-video-info"}).then(function(data) {
            $scope.canUseGPU = data["can_use_gpu"];
            $scope.columns = data['columns'];
            $scope.finishedLoading = true;
        }, function(data) {
            $scope.canUseGPU = false;
            $scope.finishedLoading = true;
        });
    };

    var initVariable = function(varName, initValue) {
        if ($scope.config[varName] == undefined) {
            $scope.config[varName] = initValue;
        }
    };

    var initVariables = function() {
        initVariable("video_name", "video.mp4");
        initVariable("detection_rate", 1);
        initVariable("detection_custom", false);
        initVariable("confidence", 0.5);
        initVariable("gpu_allocation", 0.5);
        initVariable("list_gpu", "0");
    };

    var init = function() {
        $scope.finishedLoading = false;
        initVariables();
        retrieveCanUseGPU();
    };

    init();
});
