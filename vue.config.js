module.exports = {
    pluginOptions: {
        electronBuilder: {
            builderOptions: {
                appId: "gov.sandia.srls",
                extraFiles: ["Assets"],
                artifactName: "SRLS_Setup-${version}.${ext}",
                mac: {
                    target: "default",
                    arch: ["arm64", "x64"]
                },
            },
        },
    },
};
