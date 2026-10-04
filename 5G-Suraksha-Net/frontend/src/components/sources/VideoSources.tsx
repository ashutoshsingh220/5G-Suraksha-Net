import React from 'react';
import { Camera, Video, Monitor, Film } from 'lucide-react';
import { mockVideoSources, type VideoSourceItem } from '../../data/mockData';

interface VideoSourcesProps {
  selectedSourceId?: string;
  onSelectSource?: (id: string) => void;
}

export const VideoSources: React.FC<VideoSourcesProps> = ({
  selectedSourceId = 'laptop-cam',
  onSelectSource,
}) => {
  const sources: VideoSourceItem[] = mockVideoSources;

  const handleSelect = async (id: string) => {
    if (onSelectSource) {
      onSelectSource(id);
    }
    try {
      if (id === 'drone-cam') {
        await fetch('http://localhost:8100/video/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: 'rtsp',
            url: 'rtsp://10.254.18.48:8554/drone',
          }),
        });
      } else if (id === 'laptop-cam') {
        await fetch('http://localhost:8100/video/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: 'webcam',
            camera_index: 0,
          }),
        });
      } else if (id === 'recorded-vid') {
        await fetch('http://localhost:8100/video/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: 'file',
            url: 'datasets/videos/test/synthetic_cctv.mp4',
          }),
        });
      }
    } catch {
      // Backend starting up or not reached
    }
  };

  const getSourceIcon = (type: string, isSelected: boolean) => {
    switch (type) {
      case 'WEBCAM':
        return <Monitor className={`w-3.5 h-3.5 ${isSelected ? 'text-[#3B9EFF]' : 'text-[#98A6B5]'}`} />;
      case 'MP4':
        return <Film className={`w-3.5 h-3.5 ${isSelected ? 'text-[#3B9EFF]' : 'text-[#98A6B5]'}`} />;
      default:
        return <Camera className={`w-3.5 h-3.5 ${isSelected ? 'text-[#2BC48A]' : 'text-[#98A6B5]'}`} />;
    }
  };

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="suraksha-panel-header px-3 py-1.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-1.5 text-[#E8EDF3]">
          <Video className="w-3.5 h-3.5 text-[#3B9EFF]" />
          <h2 className="text-xs font-semibold font-sans tracking-wider uppercase">
            VIDEO SOURCES
          </h2>
        </div>
        <span className="text-[11px] font-mono text-[#98A6B5] bg-[#151E28] px-2 py-0.5 rounded-[4px] border border-[#263341]">
          {sources.length} SOURCES
        </span>
      </div>

      {/* Sources List */}
      <div className="p-2 flex flex-col gap-2 overflow-y-auto">
        {sources.map((src) => {
          const isSelected = src.id === selectedSourceId;
          const isOnline = src.status === 'ONLINE';
          const isReady = src.status === 'ACTIVE';
          const isOffline = src.status === 'STANDBY';

          return (
            <button
              key={src.id}
              onClick={() => handleSelect(src.id)}
              className={`w-full flex items-center justify-between p-2 rounded-[6px] border transition-all text-left group ${
                isSelected
                  ? 'bg-[#151E28] border-[#3B9EFF] text-[#E8EDF3]'
                  : 'bg-[#151E28] border-[#263341] hover:border-[#3B9EFF]/40 hover:bg-[#151E28]/80 text-[#98A6B5]'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <div
                  className={`p-1.5 rounded-[4px] border border-[#263341] ${
                    isSelected ? 'bg-[#111821] text-[#3B9EFF]' : 'bg-[#111821] text-[#98A6B5]'
                  }`}
                >
                  {getSourceIcon(src.type, isSelected)}
                </div>
                <div>
                  <div className="text-xs font-sans font-semibold leading-none group-hover:text-[#3B9EFF] transition-colors flex items-center gap-1.5 text-[#E8EDF3]">
                    <span>{src.name}</span>
                    {isOffline && (
                      <span className="text-[10px] font-mono px-1 py-0.2 rounded-[2px] bg-[#111821] text-[#687585] border border-[#263341]">
                        OFFLINE
                      </span>
                    )}
                  </div>
                  <div className="text-[11px] font-mono text-[#98A6B5] flex items-center gap-1.5 mt-1.5 leading-none">
                    <span className="flex items-center gap-1">
                      <span
                        className={`w-1.5 h-1.5 rounded-full ${
                          isOnline
                            ? 'bg-[#2BC48A]'
                            : isReady
                            ? 'bg-[#3B9EFF]'
                            : 'bg-[#687585]'
                        }`}
                      />
                      {src.type}
                    </span>
                    <span>•</span>
                    <span>{src.resolution}</span>
                    <span>•</span>
                    <span>{src.fps} FPS</span>
                  </div>
                </div>
              </div>

              {/* Status Indicator Badge */}
              <div className="flex items-center gap-1">
                {isSelected ? (
                  <span className="text-[10px] font-mono font-medium text-[#3B9EFF] bg-[#111821] px-2 py-0.5 rounded-[4px] border border-[#3B9EFF]/40">
                    ACTIVE
                  </span>
                ) : (
                  <span className="text-[10px] font-mono text-[#687585]">
                    SELECT
                  </span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
