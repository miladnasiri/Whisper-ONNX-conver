using UnityEngine;
using UnityEngine.XR.MagicLeap;

public class AudioCaptureML2 : MonoBehaviour
{
    [SerializeField] private WhisperDecoder whisperDecoder;
    private MLAudioInput.BufferClip bufferClip;
    
    private async void Start()
    {
        // Request microphone permission
        var result = await MLPermissions.RequestPermissionAsync(MLPermission.RecordAudio);
        if (result.IsOk)
        {
            StartAudioCapture();
        }
        else
        {
            Debug.LogError("Failed to get microphone permission!");
        }
    }

    private void StartAudioCapture()
    {
        bufferClip = MLAudioInput.BufferClip.Create();
        MLAudioInput.OnBufferReady += HandleBufferReady;
        MLAudioInput.Start();
    }

    private void HandleBufferReady(MLAudioInput.Buffer buffer)
    {
        float[] audioData = new float[buffer.Samples];
        buffer.GetData(audioData);
        
        string transcription = whisperDecoder.ProcessAudio(audioData);
        Debug.Log($"Transcription: {transcription}");
    }

    private void OnDestroy()
    {
        if (MLAudioInput.IsStarted)
        {
            MLAudioInput.OnBufferReady -= HandleBufferReady;
            MLAudioInput.Stop();
        }
        bufferClip?.Destroy();
    }
}
